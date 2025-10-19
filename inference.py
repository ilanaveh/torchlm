import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import os

dataset = 'ibug_300W'  # 'ibug_300W' / 'WFLW'
img_names = {'ibug_300W': 'helen_testset_30427236_1.jpg',
             'WFLW': '0--Parade_0_Parade_marchingband_1_419x175y203.jpg'}

img_pth = os.path.join('/data', dataset, 'converted/image/test',
                       img_names[dataset])

image = cv2.imread(img_pth)[:, :, ::-1]  # BGR -> RGB
torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(
  pipnet(backbone="resnet18", pretrained=True,
         num_nb=10, num_lms=98, net_stride=32, input_size=256,
         meanface_type="wflw", map_location="cpu", checkpoint=None)
) # will auto download pretrained weights from latest release if pretrained=True
landmarks, bboxes = torchlm.runtime.forward(image)
image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)
