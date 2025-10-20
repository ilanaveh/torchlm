from torchlm.models import pipnet
import os

load_from_cp = False
choose_dataset = 'wflw'

n_lms = 98 if (choose_dataset == 'wflw') else 68
blur = 8

mdl_names = {'wflw': 'pipnet-'+choose_dataset+'-resnet18-epoch9-loss0.5262.pth',
             '300w': 'pipnet-'+choose_dataset+'-resnet18-epoch9-loss0.3619.pth'}

mdl_nm = mdl_names[choose_dataset] if load_from_cp else 'pretrained'

mdl_pth = os.path.join('./save/pipnet', mdl_nm) if load_from_cp else None

model = pipnet(backbone="resnet18", pretrained=not load_from_cp, num_nb=10, num_lms=n_lms, net_stride=32,
               input_size=256, meanface_type=choose_dataset, backbone_pretrained=not load_from_cp, checkpoint=mdl_pth)

print(f"Loaded model: {mdl_nm}")

NME, FR, AUC = model.apply_evaluating(
    annotation_path=os.path.join("./data", choose_dataset, "converted/test.txt"),
    norm_indices=[60, 72],  # the indexes of two eyeballs.
    coordinates_already_normalized=True,
    eval_normalized_coordinates=False,
    blur=blur
)
print(f"NME: {NME}, FR: {FR}, AUC: {AUC}")
