from torchlm.models import pipnet
import os


choose_dataset = 'wflw'

model = pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=98, net_stride=32,
               input_size=256, meanface_type=choose_dataset, backbone_pretrained=True)
NME, FR, AUC = model.apply_evaluating(
    annotation_path=os.path.join("./data", choose_dataset, "converted/test.txt"),
    norm_indices=[60, 72],  # the indexes of two eyeballs.
    coordinates_already_normalized=True,
    eval_normalized_coordinates=False
)
print(f"NME: {NME}, FR: {FR}, AUC: {AUC}")
