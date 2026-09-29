"""Regression tests for whole-board preservation, geometry and failure behavior."""
import unittest
import tempfile
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from board_cutout import segment_board, orient_board
import process as P


def scene(background=(231,232,215), board=(191,187,211), reflection=False):
    rgb=np.full((700,850,3),background,np.uint8)
    truth=np.zeros((700,850),np.uint8)
    points=np.array([[158,122],[691,142],[675,569],[142,550]],np.int32)
    cv2.fillConvexPoly(truth,points,255)
    rgb[truth>0]=board
    grid=rgb.copy()
    for x in range(160,700,18):cv2.line(grid,(x,110),(x,580),(120,115,133),1)
    for y in range(145,570,18):cv2.line(grid,(140,y),(705,y),(120,115,133),1)
    rgb[truth>0]=grid[truth>0]
    cv2.ellipse(rgb,(420,345),(140,115),15,0,360,(13,10,7),-1)
    # Bright labels and dark herbs must all remain part of the solid board.
    cv2.rectangle(rgb,(395,300),(445,375),(245,245,245),-1)
    if reflection:
        cv2.rectangle(rgb,(500,155),(555,460),(255,255,255),-1)
    for i in range(9):
        cv2.rectangle(rgb,(177,190+i*28),(198,210+i*28),[(220,20,20),(20,200,20),(20,20,220)][i%3],-1)
    return rgb,truth


class BoardTests(unittest.TestCase):
    def assert_complete(self,rgb,truth):
        mask,bbox,report=segment_board(Image.fromarray(rgb))
        inside=cv2.erode(truth,np.ones((7,7),np.uint8))>0
        overlap=np.sum((mask>=127)&(truth>0))
        iou=overlap/np.sum((mask>=127)|(truth>0))
        self.assertGreater(iou,.96)
        self.assertTrue(np.all(mask[inside]==255),'Board interior must have no cutout holes')
        self.assertEqual(report['detector'],'board_boundary')
        return mask
    def test_dark_herbs_grid_and_bright_label_preserved(self):
        self.assert_complete(*scene())
    def test_all_quarter_turns_preserve_board(self):
        rgb,truth=scene()
        for k in (1,2,3):
            with self.subTest(turn=k):self.assert_complete(np.rot90(rgb,k).copy(),np.rot90(truth,k).copy())
    def test_arbitrary_rotation(self):
        rgb,truth=scene();matrix=cv2.getRotationMatrix2D((425,350),29,1)
        rgb=cv2.warpAffine(rgb,matrix,(850,700),borderValue=(231,232,215))
        truth=cv2.warpAffine(truth,matrix,(850,700),flags=cv2.INTER_NEAREST)
        self.assert_complete(rgb,truth)
    def test_reflection_not_cut_out(self):
        self.assert_complete(*scene(reflection=True))
    def test_gray_board_on_dark_background(self):
        self.assert_complete(*scene(background=(25,25,25),board=(187,187,187)))
    def test_plain_image_rejected(self):
        for color in ('white','black','gray'):
            with self.subTest(color=color),self.assertRaises(ValueError):
                segment_board(Image.new('RGB',(500,500),color))
    def test_unrelated_round_herb_is_not_board(self):
        rgb=np.full((600,600,3),230,np.uint8);cv2.circle(rgb,(300,300),150,(20,20,20),-1)
        with self.assertRaises(ValueError):segment_board(Image.fromarray(rgb))
    def test_real_batch_branch_fixed_size_and_report(self):
        rgb,_=scene()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'input';source.mkdir();Image.fromarray(rgb).save(source/'ruler.png')
            for size,kb in ((1200,3072),(800,2000)):
                target=root/str(size)
                self.assertEqual(P.do_cutout(source,target,method='board',size=size,max_kb=kb,save_report=True),(1,0))
                output=target/'ruler.jpg'
                with Image.open(output) as im:
                    self.assertEqual(im.size,(size,size))
                    self.assertTrue(np.all(np.asarray(im)[:20]>248))
                self.assertLessEqual(output.stat().st_size,kb*1024)
                report=json.loads((target/'处理报告.json').read_text(encoding='utf-8'))
                self.assertEqual(report[0]['detector'],'board_boundary')

class OrientationTests(unittest.TestCase):
    def test_quarter_turns_and_arbitrary_tilt_all_put_chart_left(self):
        rgb,truth=scene()
        variants=[np.rot90(rgb,k).copy() for k in range(4)]
        matrix=cv2.getRotationMatrix2D((425,350),29,1)
        variants.append(cv2.warpAffine(rgb,matrix,(850,700),borderValue=(231,232,215)))
        for i,arr in enumerate(variants):
            with self.subTest(variant=i):
                im=Image.fromarray(arr);mask,_,report=segment_board(im)
                out,alpha,meta=orient_board(im,mask,report)
                hsv=cv2.cvtColor(np.array(out),cv2.COLOR_RGB2HSV)
                vivid=(hsv[:,:,1]>150)&(hsv[:,:,2]>100)&(alpha>240)
                ys,xs=np.where(vivid)
                by,bx=np.where(alpha>127)
                self.assertGreater(len(xs),1000)
                self.assertLess(np.quantile(xs,.95),bx.min()+.18*(bx.max()-bx.min()))
                # A perspective trapezoid can retain a small chart slant; bound angle, not correlation.
                self.assertLess(abs(np.polyfit(ys,xs,1)[0]),np.tan(np.deg2rad(4)))
                self.assertGreater(np.sum(alpha>127)/np.sum(mask>127),.995)
                self.assertLess(np.sum(alpha>127)/np.sum(mask>127),1.005)
                self.assertEqual(meta['orientation'],'upright_color_chart_left')
    def test_missing_color_chart_fails_instead_of_guessing(self):
        rgb,_=scene();rgb[180:460,170:205]=(190,190,210)
        im=Image.fromarray(rgb);mask,_,report=segment_board(im)
        with self.assertRaisesRegex(ValueError,'朝向'):
            orient_board(im,mask,report)
    def test_ambiguous_opposite_charts_fail(self):
        rgb,_=scene()
        for i in range(9):
            cv2.rectangle(rgb,(638,190+i*28),(659,210+i*28),[(220,20,20),(20,200,20),(20,20,220)][i%3],-1)
        im=Image.fromarray(rgb);mask,_,report=segment_board(im)
        with self.assertRaisesRegex(ValueError,'朝向'):
            orient_board(im,mask,report)

if __name__=='__main__':unittest.main()
