"""显式下载本地模型；不上传任何产品图片。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
from rembg import new_session
if __name__ == '__main__':
    print('正在安装/检查 BiRefNet Lite，首次需要下载约 224 MB。')
    new_session('birefnet-general-lite', providers=['CPUExecutionProvider'])
    print('模型已就绪，可以运行 开始.bat。')
