import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules import conv
from ultralytics.utils import yaml_load

#import your custom modules

from dfem_modules import TuaBottleneck, Scalseq, Zoomcat

#hack: register modules into ultralytics' global namespace so the YAML parser finds them

import ultralytics.nn.modules.block   
ultralytics.nn.modules.block.TuaBottleneck = TuaBottleneck
#we map them to generic places or just inject them into the module list
setattr(ultralytics.nn.modules.block, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.modules.block, 'Scalseq', Scalseq)
setattr(ultralytics.nn.modules.block, 'Zoomcat', Zoomcat)

def run_full_dfem_net():
    print("Building Full DFEM-Net Model")
    try:
        model = YOLO("dfem_net.yaml")

        print("SUCCESS: Model architecture parsed!")
        print(model.info())

        print("Running forward pass(CPU)")
        img = torch.randn(1, 3, 640, 640)
        results = model(img)

        print("Forward pass successful. Detections output shape:", results[0].boxes.shape)

    except Exception as e:
        print("f\nCRASHED: {e}")
        print("Tip: Ensure 'dfem_net.yaml' is in the same folder and contains valid Python syntax.")

if __name__ == "__main__":
    run_full_dfem_net()

