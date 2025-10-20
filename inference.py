import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import os
from PIL import Image

save_figs = False
save_dir = '/save/inference'

choose_dataset = '300W'  # '300W' / 'wflw' [dataset from which test image is taken]
choose_mdl_trainset = 'wflw'  # 'wflw' / '300w'  [dataset on which model was trained on]
n_lms = 98 if (choose_mdl_trainset == 'wflw') else 68
img_names = {'wflw': '0--Parade_0_Parade_marchingband_1_419x175y203.jpg',
             '300w': 'helen_testset_30427236_1.jpg'}
mdl_names = {'wflw': 'pipnet-'+choose_mdl_trainset+'-resnet18-epoch9-loss0.5262.pth',
             '300w': 'pipnet-'+choose_mdl_trainset+'-resnet18-epoch9-loss0.3619.pth'}

img_pth = os.path.join('./data', choose_dataset, 'converted/image/test',
                       img_names[choose_dataset])
mdl_pth = os.path.join('./save/pipnet', mdl_names[choose_mdl_trainset])

image = cv2.imread(img_pth)[:, :, ::-1]  # BGR -> RGB
torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(
  pipnet(backbone="resnet18", pretrained=False,
         num_nb=10, num_lms=n_lms, net_stride=32, input_size=256,
         meanface_type=choose_mdl_trainset, map_location="cpu", checkpoint=mdl_pth)
)  # will auto download pretrained weights from latest release if pretrained=True
landmarks, bboxes = torchlm.runtime.forward(image)
image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)

if save_figs:
    im2save = Image.fromarray(image)
    im2save.save(os.path.join(save_dir, f'test_im_from_{choose_dataset}_mdl_trained_on_{choose_mdl_trainset}.jpg'))
