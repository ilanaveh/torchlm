import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import os
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import numpy as np


load_from_cp = True
blur = 6
save_figs = True
save_dir = './save/inference'

choose_dataset = '300w'  # '300w' / 'wflw' [dataset from which test image is taken]
choose_mdl_trainset = 'wflw'  # 'wflw' / '300w'  [dataset on which model was trained on]
n_lms = 98 if (choose_mdl_trainset == 'wflw') else 68
img_names = {'wflw': '0--Parade_0_Parade_marchingband_1_419x175y203.jpg',
             '300w': 'helen_testset_30427236_1.jpg'}
mdl_names = {'wflw': 'pipnet-'+choose_mdl_trainset+'-resnet18-epoch9-loss0.5262.pth',
             '300w': 'pipnet-'+choose_mdl_trainset+'-resnet18-epoch9-loss0.3619.pth'}

img_pth = os.path.join('./data', choose_dataset, 'converted/image/test',
                       img_names[choose_dataset])
mdl_pth = os.path.join('./save/pipnet', mdl_names[choose_mdl_trainset]) if load_from_cp else None

image = cv2.imread(img_pth)[:, :, ::-1]  # BGR -> RGB


class GaussianBlur(object):
    """Apply Gaussian blur filter with the given sigma to the input PIL Image.
    Args:
        sigma (int): Desired Gaussian blur level sigma

    Taken from: W:\dannyh\work\code\PyTorch\vggface2_lookdir\datasets\custom_transforms.
   """

    def __init__(self, sigma):
        assert isinstance(sigma, int)
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


if blur:
    transform = transforms.Compose([transforms.ToPILImage(), GaussianBlur(sigma=blur)])
    image = transform(image)
    image = np.asarray(image)

torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(
  pipnet(backbone="resnet18", pretrained=not load_from_cp,
         num_nb=10, num_lms=n_lms, net_stride=32, input_size=256,
         meanface_type=choose_mdl_trainset, map_location="cpu", checkpoint=mdl_pth)
)  # will auto download pretrained weights from latest release if pretrained=True
landmarks, bboxes = torchlm.runtime.forward(image)
image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)

if save_figs:
    im2save = Image.fromarray(image)
    mdl_suf = f'trained_on_{choose_mdl_trainset}' if load_from_cp else 'pretrained'
    blur_suf = f'_blur{blur}' if blur else ''
    im_save_nm = f'test_im_from_{choose_dataset}{blur_suf}_mdl_{mdl_suf}.jpg'

    im2save.save(os.path.join(save_dir, im_save_nm))

