from torchlm.models import pipnet
from torchlm.data import LandmarksWFLWConverter, Landmarks300WConverter
import os

num_lms = {
    'wflw': 98,
    '300w': 68
}

choose_ds = '300w'  # 'wflw' / '300w'
freeze_backbone = False

# ~~~~~~~~~~~~~~~~~~~~
# ~~~ Convert data ~~~
# ~~~~~~~~~~~~~~~~~~~~

# setup your path to the original downloaded dataset from official
# 1. WFLW:
wflw_save_dir = "./data/wflw/converted"
if not os.path.exists(wflw_save_dir) or not os.listdir(wflw_save_dir):
    converter = LandmarksWFLWConverter(
        data_dir="../../data/WFLW", save_dir=wflw_save_dir,
        extend=0.2, rebuild=True, target_size=256, keep_aspect=False,
        force_normalize=True, force_absolute_path=True
    )
    converter.convert()
    converter.show(count=30)  # show you some converted images with landmarks for debugging

# 2. 300W:
ibug_300w_save_dir = "./data/300w/converted"
if not os.path.exists(ibug_300w_save_dir) or not os.listdir(ibug_300w_save_dir):
    converter = Landmarks300WConverter(
        data_dir="../../data/ibug_300W_large_face_landmark_dataset", save_dir=ibug_300w_save_dir,
        extend=0.2, rebuild=True, target_size=256, keep_aspect=False,
        force_normalize=True, force_absolute_path=True
    )
    converter.convert()
    converter.show(count=30)  # show you some converted images with landmarks for debugging

# ~~~~~~~~~~~~~~~~~~
# ~~~ Load model ~~~
# ~~~~~~~~~~~~~~~~~~
model = pipnet(backbone="resnet18", pretrained=False, num_nb=10, num_lms=num_lms[choose_ds],
               net_stride=32, input_size=256, meanface_type=choose_ds, backbone_pretrained=True)

# ~~~~~~~~~~~~~~~~~~~
# ~~~ Train model ~~~
# ~~~~~~~~~~~~~~~~~~~
model.apply_freezing(backbone=freeze_backbone)
model.apply_training(
    annotation_path=os.path.join('./data', choose_ds, 'converted/train.txt'),
    # or fine-tuning your custom data
    num_epochs=10,
    learning_rate=0.0001,
    save_dir="./save/pipnet",
    save_prefix="pipnet-" + choose_ds + "-resnet18",
    save_interval=10,
    logging_interval=1,
    device="cuda",
    coordinates_already_normalized=True,
    batch_size=16,
    num_workers=4,
    shuffle=True
)
