import unittest, tempfile, io, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from PIL import Image
import product_cutout as C
import process as P

class CutoutTests(unittest.TestCase):
    def test_holes_filled_but_concave_bag_edge_retained(self):
        a=np.zeros((180,140),np.uint8); a[15:165,20:120]=255
        a[60:120,45:100]=0; a[40:75,20:30]=0; a[2:4,2:4]=255
        cleaned,_=C.clean_product_mask(a)
        self.assertEqual(cleaned[90,65],255)
        self.assertEqual(cleaned[55,22],0)
        self.assertEqual(cleaned[2,2],0)
    def test_no_fill_mode_preserves_real_holes(self):
        a=np.zeros((100,100),np.uint8); a[10:90,10:90]=255; a[30:60,30:60]=0
        cleaned,_=C.clean_product_mask(a,False)
        self.assertEqual(cleaned[45,45],0)
    def test_pale_thin_tip_not_cropped_by_color(self):
        a=np.zeros((100,100),np.uint8); a[20:85,25:75]=255; a[5:25,49:51]=255
        im=Image.new('RGB',(100,100),(250,250,250))
        out=C.compose_square(im,a,size=200,pad=10)
        self.assertEqual(out.size,(200,200))
        arr=np.array(out)
        self.assertTrue(np.any(arr[10:20]<255))
        self.assertTrue(np.all(arr[:8]==255))
    def test_guidance_does_not_punch_holes(self):
        a=np.zeros((120,120),np.uint8); a[10:110,10:110]=255
        rgb=np.random.default_rng(4).integers(0,255,(120,120,3),dtype=np.uint8)
        out=C.guided_alpha(rgb,a)
        self.assertTrue(np.all(out[20:100,20:100]==255))
        self.assertTrue(np.all(out[:3]==0))
    def test_bbox_includes_last_pixel(self):
        a=np.zeros((9,8),np.uint8); a[8,7]=255
        self.assertEqual(C.mask_bbox(a),(7,8,1,1))
    def test_fixed_size_and_byte_limit(self):
        im=Image.new('RGB',(800,800),'white')
        data=C.encode_square(im,2000000)
        self.assertLessEqual(len(data),2000000)
        self.assertEqual(Image.open(io.BytesIO(data)).size,(800,800))
        with self.assertRaises(ValueError):C.encode_square(im,10)
    def test_orientation_before_conversion(self):
        im=Image.new('RGB',(60,30),'red'); exif=im.getexif(); exif[274]=6
        b=io.BytesIO(); im.save(b,'JPEG',exif=exif); b.seek(0)
        self.assertEqual(P.rotate(Image.open(b)).size,(30,60))
    def test_root_and_nested_images_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/'input'; src.mkdir(); (src/'batch'/'nested').mkdir(parents=True)
            for rel in ['root.png','batch/nested/child.png']:
                Image.new('RGB',(100,100),'green').save(src/rel)
            out=src/'已完成'; out.mkdir(); Image.new('RGB',(10,10)).save(out/'old.jpg')
            def fake(im,**kw):
                a=np.zeros((im.height,im.width),np.uint8); a[10:-10,10:-10]=255
                return a,C.mask_bbox(a),{'warnings':[],'model':'test'}
            with patch.object(P,'segment_product',side_effect=fake):
                self.assertEqual(P.do_cutout(src,out,method='bag',size=800,max_kb=2000),(2,0))
            self.assertTrue((out/'root.jpg').exists())
            self.assertTrue((out/'batch/nested/child.jpg').exists())
            self.assertFalse((out/'处理报告.json').exists())
            self.assertTrue(all(f.suffix == '.jpg' for f in out.rglob('*') if f.is_file()))
    def test_missing_model_does_not_download(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'U2NET_HOME':td}):
            C.model_session.cache_clear()
            with patch('rembg.new_session') as new:
                with self.assertRaises(RuntimeError): C.model_session('birefnet-general-lite')
                new.assert_not_called()
    def test_model_failure_reported_without_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/'in'; src.mkdir(); Image.new('RGB',(80,80)).save(src/'a.png')
            with patch.object(P,'segment_product',side_effect=RuntimeError('model missing')):
                self.assertEqual(P.do_cutout(src,root/'out',method='bag'),(0,1))
            self.assertFalse((root/'out'/'a.jpg').exists())

    def test_fragmented_prediction_retries_with_context(self):
        initial=np.zeros((100,100),np.uint8)
        initial[15:35,20:80]=255; initial[50:75,40:60]=255
        retry=np.zeros((130,130),np.uint8); retry[25:105,30:100]=255
        with patch.object(C,'model_session',return_value=object()), patch('rembg.remove',side_effect=[Image.fromarray(initial),Image.fromarray(retry)]) as predict:
            mask,bbox,report=C.segment_product(Image.new('RGB',(100,100),'gray'))
        self.assertEqual(predict.call_count,2)
        self.assertTrue(report['context_retry'])
        self.assertGreater(report['coverage'],0.5)
        self.assertEqual(mask[70,30],255)
        self.assertTrue(report['warnings'])

    def test_retry_cannot_replace_product_with_unrelated_region(self):
        initial=np.zeros((100,100),np.uint8); initial[10:35,10:35]=255
        retry=np.zeros((130,130),np.uint8); retry[60:120,60:120]=255
        with patch.object(C,'model_session',return_value=object()), patch('rembg.remove',side_effect=[Image.fromarray(initial),Image.fromarray(retry)]):
            mask,bbox,report=C.segment_product(Image.new('RGB',(100,100),'gray'))
        self.assertFalse(report['context_retry'])
        self.assertEqual(mask[20,20],255)
        self.assertEqual(mask[70,70],0)

if __name__=='__main__':unittest.main()
