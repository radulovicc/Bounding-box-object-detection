"""
SSD (Single Shot MultiBox Detector) sa MobileNetV3 backbone-om.

Ova implementacija koristi torchvision-ov gotovi SSD model,
ali sa PRETRAINED=False (random inicijalizacija tezina).

Arhitektura:
  - Backbone: MobileNetV3 (mali, ~2.5M parametara)
  - Detection head: SSD head sa vise anchor box-ova po poziciji
  - Ulaz: slike dimenzije 320x320
  - Izlaz: bounding box-ovi + klase

Reference:
  - SSD: https://arxiv.org/abs/1512.02325
  - MobileNetV3: https://arxiv.org/abs/1905.02244
"""

import torch
import torch.nn as nn
import torchvision
from typing import Dict, List, Optional


class SSDMobileNetDetector(nn.Module):
    """
    SSD detektor za detekciju vozila i pesaka.

    Podrzava trening od nule (random inicijalizacija) ili
    koriscenje pretreniranog backbone-a (opciono za poredjenje).

    Args:
        num_classes: Broj klasa (ukljucujuci background = 1)
        pretrained_backbone: Da li koristiti pretrenirani backbone
                            (False = random inicijalizacija)
        pretrained: Da li koristiti potpuno pretreniran model
                   (samo za benchmark/poredjenje)
    """

    def __init__(
        self,
        num_classes: int = 4,  # background + Car + Pedestrian + Bicycle
        pretrained_backbone: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained_backbone = pretrained_backbone
        self.pretrained = pretrained

        # Kreiramo model
        if pretrained:
            # Potpuno pretreniran model (samo za benchmark/poredjenje)
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                weights="DEFAULT",
                num_classes=num_classes,
            )
            print("[INFO] Koristim POTPUNO pretreniran model (samo za benchmark)")
        else:
            # Model sa random inicijalizacijom (ZA PRAVI TRENING)
            weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None

            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
            )

            if pretrained_backbone:
                print("[INFO] Koristim pretreniran backbone + random detection head")
            else:
                print("[INFO] Model sa POTPUNO RANDOM inicijalizacijom tezina!")

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ):
        """
        Prolaz kroz model.

        U trening modu vraca recnik gubitaka.
        U evaluacionom modu vraca listu predikcija.

        Args:
            images: Lista tenzora slika [C, H, W] (normalizovane na [0,1])
            targets: Lista dict-ova sa 'boxes' i 'labels' (samo za trening)

        Returns:
            - Trening: {'classification': loss, 'bbox_regression': loss}
            - Evaluacija: [{'boxes': tensor, 'labels': tensor, 'scores': tensor}, ...]
        """
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            predictions = self.model(images)
            return predictions


def create_model(
    num_classes: int = 4,
    from_scratch: bool = True,
) -> SSDMobileNetDetector:
    """
    Factory funkcija za kreiranje modela.

    Args:
        num_classes: Broj klasa (background + Car + Pedestrian + Bicycle)
        from_scratch: True = random inicijalizacija (za pravi trening)
                     False = pretreniran backbone (za poredjenje)

    Returns:
        SSDMobileNetDetector instanca
    """
    return SSDMobileNetDetector(
        num_classes=num_classes,
        pretrained_backbone=not from_scratch,
        pretrained=False,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testiranje SSDMobileNetDetector modela")
    print("=" * 60)

    # Test 1: Model sa random inicijalizacijom
    print("\n[TEST 1] Kreiranje modela sa RANDOM tezinama...")
    model = create_model(num_classes=4, from_scratch=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Ukupno parametara: {total_params:,}")
    print(f"    Trenirajuci parametri: {trainable_params:,}")

    # Test 2: Forward pass sa laznim podacima
    print("\n[TEST 2] Test forward pass-a (TRAINING mod)...")
    # Koristimo 2 slike (BatchNorm zahteva batch_size >= 2)
    dummy_images = [torch.randn(3, 320, 320), torch.randn(3, 320, 320)]
    model.train()

    dummy_targets = [
        {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
        },
        {
            'boxes': torch.tensor([[30, 40, 120, 130]], dtype=torch.float32),
            'labels': torch.tensor([2], dtype=torch.int64),
        },
    ]

    with torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        losses = model(dummy_images, dummy_targets)

    print(f"    Classification loss: {losses['classification'].item():.4f}")
    print(f"    BBox regression loss: {losses['bbox_regression'].item():.4f}")
    print(f"    Ukupan gubitak: {sum(losses.values()).item():.4f}")

    # Test 3: Inference mod
    print("\n[TEST 3] Test inference mod-a (EVAL)...")
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_images)

    print(f"    Broj detekcija: {len(predictions[0]['boxes'])}")
    print(f"    Confidence skorovi: {predictions[0]['scores'][:5].tolist()}")

    # Test 4: GPU
    if torch.cuda.is_available():
        print("\n[TEST 4] GPU test...")
        model = model.cuda()
        dummy_images_gpu = [img.cuda() for img in dummy_images]
        dummy_targets_gpu = [
            {k: v.cuda() if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in dummy_targets
        ]
        model.train()
        losses = model(dummy_images_gpu, dummy_targets_gpu)
        print(f"    GPU gubitak: {sum(losses.values()).item():.4f}")

    print("\n" + "=" * 60)
    print("Svi testovi su prosli uspesno!")
    print("=" * 60)
