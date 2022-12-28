# NanoNet: Real-Time Polyp Segmentation in Video Capsule Endoscopy and Colonoscopy

This is an unofficial PyTorch implementation of [NanoNet](https://github.com/DebeshJha/NanoNet).


## Requirements

```
pip install -r requirements.txt
```


## Model


<div align="center">

<img src="https://raw.githubusercontent.com/DebeshJha/NanoNet/main/figures/nanonet.png" width="400">

</div>

The architecture of the NanoNet follows an encoder-decoder approach as shown in Figure 1. As depicted in Figure below, the network architecture uses a pre-trained model as an encoder, followed by the three decoder blocks. Using pre-trained ImageNet models for transfer learning has become the best choice for many CNN architectures. It helps the model converge much faster and achieve high-performance compared to the non-pre-trained model. The proposed architecture uses a MobileNetV2 model pre-trained on the ImageNet dataset as the encoder. The decoder is built using a modified version of the residual block, which was initially introduced by He et al. The encoder is used to capture the required contextual information from the input, whereas the decoder is used to generate the final output by using the contextual information extracted by the encoder.
