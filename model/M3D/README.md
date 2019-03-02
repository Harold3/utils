## M3D <br>
- Torch impelment of paper:<br>
[Multi-scale 3D Convolution Network for Video Based Person Re-Identification](https://arxiv.org/abs/1811.07468)
# Introduction
- The whole network in paper is a two-stream convolution network to extract spatial and temporal 
  cues for video based person ReID.<br>
- M3D layer is used in temporal stream which is capable of learning multi-scale temporal feature.The whole idea is inspired by dilation in 2d convolution,and it combines 3 kernel with different
  dilation in time dimension.
