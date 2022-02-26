import paddle
import paddle.vision.transforms as T
import numpy as np
from config import get
from PIL import Image

__all__ = ['MicroExpression']

# 定义图像的大小
image_shape = get('image_shape')
IMAGE_SIZE = (image_shape[1], image_shape[2])


class MicroExpression(paddle.io.Dataset):
    """
    微表情数据集类的定义
    """

    def __init__(self, mode='train'):
        """
        初始化函数
        """
        assert mode in ['train', 'test', 'valid'], 'mode is one of train, test, valid.'

        self.data = []

        with open('classification2/{}.txt'.format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split('\t')

                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

        if mode == 'train':
            self.transforms = T.Compose([
                T.RandomResizedCrop(IMAGE_SIZE),    # 随机裁剪大小
                T.RandomHorizontalFlip(0.5),        # 随机水平翻转
                T.ToTensor(),                       # 数据的格式转换和标准化 HWC => CHW  
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),                 # 图像大小修改
                T.RandomCrop(IMAGE_SIZE),      # 随机裁剪
                T.ToTensor(),                  # 数据的格式转换和标准化 HWC => CHW
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 图像归一化
            ])
        
    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transforms(image)

        return image, np.array(label, dtype='int64')

    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)
