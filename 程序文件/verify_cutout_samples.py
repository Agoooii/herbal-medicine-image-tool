"""Run real bag samples without modifying originals or production output folders."""
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import process as P
from product_cutout import compose_square, encode_square

B=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent/'验证资料'

def verify_and_preview():
    src=O/'样图'
    dest=O/'新版成品'
    records=json.loads((dest/'处理报告.json').read_text(encoding='utf-8'))
    assert len(records)==5 and all(r['status'] in ('ok','review') for r in records)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    sheet=Image.new('RGB',(1200,430*len(records)),(240,241,244))
    rows=[]
    for i,r in enumerate(records):
        p=src/r['source']; new=dest/r['output']
        with Image.open(new) as im:
            assert im.size==(1200,1200) and new.stat().st_size<=3072*1024
            assert np.all(np.asarray(im)[0,:]>245), 'Top margin must be white'
        original=ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        work=original.copy(); work.thumbnail((2400,2400))
        mask=np.asarray(Image.open(dest/(p.stem+'_mask.png')).convert('L'))
        out800=compose_square(work,mask,size=800,pad=30)
        data=encode_square(out800,2000*1024)
        outdir=O/'800像素成品'; outdir.mkdir(exist_ok=True)
        (outdir/(p.stem+'.jpg')).write_bytes(data)
        assert Image.open(outdir/(p.stem+'.jpg')).size==(800,800)
        for j,(label,im) in enumerate([('原图',original),('旧版',Image.open(O/(p.stem+'_旧版.jpg'))),('新版',Image.open(new))]):
            tile=ImageOps.contain(im.convert('RGB'),(385,385))
            sheet.paste(tile,(400*j+(400-tile.width)//2,i*430+40+(385-tile.height)//2))
            ImageDraw.Draw(sheet).text((400*j+12,i*430+7),p.stem+' / '+label,font=font,fill='#202020')
        rows.append({'source':r['source'],'status':r['status'],'bytes1200':new.stat().st_size,'bytes800':len(data),'warnings':r['warnings']})
    sheet.save(O/'前后对比.jpg',quality=90)
    (O/'验证结果.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    if '--preview-only' not in sys.argv:
        counts=P.do_cutout(O/'样图',O/'新版成品',method='bag',erode=0,feather=0,save_mask=True,save_report=True)
        assert counts==(5,0),counts
    verify_and_preview()
