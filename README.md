# Deploy yolov-pose TensorRT for Windows

## References

Official yolov7 implement: https://github.com/WongKinYiu/yolov7

Official yolov7-pose implement: https://github.com/WongKinYiu/yolov7/tree/pose

Deploy Yolov7-pose with TensorRT for Linux: https://github.com/nanmi/yolov7-pose

# Build project form references
## Installation

  install torch and related package

  ```shell
  # python 3.7 (or higher) is recommended

  pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

  pip3 install -r requirements.txt
  
  # Install onnxsim
  pip3 install onnxsim
  ```

## Build projects on Windows

Based on Deploy [Yolov7-pose with TensorRT for Linux](https://github.com/nanmi/yolov7-pose), we change congifurations to build on Windows

### YoloLayer_TRT_v7.0: generate plugin library to build tensorRT engine

  - Change CMakeLists.txt to build on Wondows

    ```shell
    ...
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
    ...
    set(TENSORRT_INCLUDE_DIR D:/Library/TensorRT-8.2.4.2/include/)
    set(TENSORRT_LIBRARY_DIR D:/Library/TensorRT-8.2.4.2/lib/)
    ```
    
  - Use CMake (cmake-gui) on Windows to build project, fix paths to Generate
  - Open ".sln" file with Visual Studio, right click for project on "Solution Explorer" tab to change "Build Dependencies / Build Customizations" to CUDA 11.x
  - Open yolo project Properties, change Target Name to yololayer
  - Right click on "yololayer.cu", in tab "General", change Item Type to "CUDA C/C++"
  - Make sure paths and name for libraries on Linker are correct
  
    ```shell
    cudart.lib;cublas.lib;cudnn.lib;cudnn64_8.lib;nvinfer.lib;nvinfer_plugin.lib;nvonnxparser.lib;nvparsers.lib
    ```
    
  - In tab "CUDA Linker" of project Properties, additional Options: -API_EXPORTS
  - Remove "Object Files" of yolo project on "Solution Explorer" tab
  - Build project, get "yololayer.dll" and "yololayer.lib" in "YoloLayer_TRT_v7.0/build/Release" folder
  
### Build TensorRT engine by using generated plugin

  - Download [Official yolov7 implement](https://github.com/WongKinYiu/yolov7), [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt), add following code to make "export_pose.py"
  
    ```shell
    import sys
    sys.path.append('./')  # to run '$ python *.py' files in subdirectories
    import torch
    import torch.nn as nn
    import models
    from models.experimental import attempt_load
    from utils.activations import Hardswish, SiLU

    # Load PyTorch model
    weights = 'yolov7-w6-pose.pt'
    device = torch.device('cuda:0')
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = True # set Detect() layer grid export
    model.eval()

    # Input
    img = torch.randn(1, 3, 960, 960).to(device)  # image size(1,3,320,192) iDetection
    torch.onnx.export(model, img, 'yolov7-w6-pose.onnx', verbose=False, opset_version=12, input_names=['images'])
    ```
  
  - Generate "yolov7-w6-pose.onnx" model by using "export_pose.py"
  - Generate "yolov7-w6-pose-sim.onnx" model by this command:
 
    ```shell
    onnxsim yolov7-w6-pose.onnx yolov7-w6-pose-sim.onnx
    ```
  
  - **NOTICE for Tensor Name**: tensors names in file "add_custom_yolo_op.py" are names of old version of "yolov7-w6-pose.pt", to check new tensors name:
    
    - Open [netron.app](https://netron.app/), open Model "yolov7-w6-pose-sim.onnx"
    - "Ctrl + F" to search "Transpose" layers, there will be 4 Transpose layers

      <img src="./transpose.png" title="" alt="" width="80%" height="80%"></img>
      
    - For each Transpose layer, find "concat" layer above it to check "outputs" name
    
      <img src="./concat_output.png" title="" alt="" width="80%" height="80%"></img>
    
    - "Concat" output names are used for tensors names in "add_custom_yolo_op.py"

      ```shell
      inputs = [tensors["745"].to_variable(dtype=np.float32), 
      tensors["802"].to_variable(dtype=np.float32),
      tensors["859"].to_variable(dtype=np.float32),
      tensors["916"].to_variable(dtype=np.float32)]
      ```
    - Generate "yolov7-w6-pose-sim-yolo.onnx" model by using "add_custom_yolo_op.py"
    
 Updating ....
