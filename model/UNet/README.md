# Introduction    
The U-Net is a widly used convolutional encoder-decoder neural network,
especially in semantic segmentaition task and GAN's generator.
## Modifications
This code's modifications to the original paper:<br>
- padding is used in 3x3 convolutions to prevent loss of border pixels<br>
- merging outputs does not require cropping due to (1)<br>
- residual connections can be used by specifying UNet(merge_mode='add')<br>
- if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'),<br>
        then an additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality<br> by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution <br>(specified by upmode='transpose')
