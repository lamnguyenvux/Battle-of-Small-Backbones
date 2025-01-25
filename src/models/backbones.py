import torch
import torch.nn as nn
import timm

from wavemix.classification import WaveMix
# from src.models.repvgg import create_RepVGG_A0

map_model = {
    'convnext-nano': 'convnext_nano.in12k_ft_in1k',  # 15.6 M
    'convnext-tiny': 'convnext_tiny.in12k_ft_in1k',  # 28.6 M
    'efficientnetv2': 'efficientnetv2_rw_s.ra2_in1k',  # 24.1 M
    'regnet': 'regnety_032.ra_in1k',  # 19.4M
    'repvgg-a1': 'repvgg_a1',  # 14.1 M
    'repvgg-a2': 'repvgg_a2.rvgg_in1k',  # 28.2 M
    'repvgg-a0': 'repvgg_a0',  # 9.1 M
    'ghostnet': 'ghostnet_100.in1k',  # 5.2 M
    'mobilenetv4': 'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',  # 11.1 M
    'repghostnet': 'repghostnet_200.in1k', # 9.8 M
    'ghostnetv2': 'ghostnetv2_160.in1k' # 12.4 M
}


def get_model(model_name: str, num_classes: int = 2):
    if model_name == "wavemix":
        model = WaveMix(
            num_classes=1000,
            depth=16,
            mult=2,
            ff_channel=192,
            final_dim=192,
            dropout=0.5,
            level=3,
            initial_conv='pachify',
            patch_size=4
        )

        url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemix_192_16_75.06.pth'
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(url, map_location="cpu"))
        model.pool[2] = nn.Linear(192, num_classes)

    elif model_name == "shufflenet":
        from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
        model = shufflenet_v2_x2_0(
            weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, num_classes)

    elif model_name == "resnet":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, num_classes)

    else:
        name = map_model.get(model_name, None)
        if name is None:
            name = model_name
        model = timm.create_model(
            model_name=name,
            num_classes=num_classes,
            pretrained=True
        )

    # elif model_name == "swin":
    #     from torchvision.models import swin_t, Swin_T_Weights
    #     model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    #     model.head = nn.Linear(768, num_classes)

    # elif model_name == "swinv2":
    #     from torchvision.models import swin_v2_t, Swin_V2_T_Weights
    #     model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    #     model.head = nn.Linear(768, num_classes)

    # elif model_name == "efficientnet":
    #     from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    #     model = efficientnet_v2_s(
    #         weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    #     model.classifier[1] = nn.Linear(1280, num_classes)

    # elif model_name == "convnext":
    #     from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    #     model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    #     model.classifier[2] = nn.Linear(768, num_classes)

    # elif model_name == "densenet":
    #     from torchvision.models import densenet161, DenseNet161_Weights
    #     model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
    #     model.classifier = nn.Linear(2208, num_classes)

    # elif model_name == "inception":
    #     from torchvision.models import inception_v3, Inception_V3_Weights
    #     model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    #     model.fc = nn.Linear(2048, num_classes)

    # elif model_name == "mobilenet":
    #     from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    #     model = mobilenet_v3_large(
    #         weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    #     model.classifier[3] = nn.Linear(1280, num_classes)

    # elif model_name == "regnet":
    #     from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
    #     model = regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
    #     model.fc = nn.Linear(1512, num_classes)

    # elif model_name == "resnext":
    #     from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
    #     model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    #     model.fc = nn.Linear(2048, num_classes)

    # elif model_name == "shufflenet":
    #     from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
    #     model = shufflenet_v2_x2_0(
    #         weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    #     model.fc = nn.Linear(2048, num_classes)
    # elif model_name == "repvgg":
    #     model = create_RepVGG_A0(
    #         num_classes=num_classes
    #     ).load_state_dict(torch.load("", map_location="cpu"))
    # else:
    #     raise ValueError("Model {} is not found")

    return model
