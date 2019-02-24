import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, out_channels, normal_layer=nn.InstanceNorm2d,
                 activation=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3),
                 normal_layer(out_channels), activation]
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3),
                  normal_layer(out_channels)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=9, ngf=64, n_down=4, normal_layer=nn.InstanceNorm2d,
                 activation=nn.ReLU(True)):
        super(GlobalGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
                 normal_layer, activation]
        for i in range(n_down):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, stride=2, kernel_size=3, padding=1),
                      normal_layer, activation]

        mult = 2 ** n_down
        for i in range(n_blocks):
            model += [ResBlock(ngf * mult)]

        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult / 2, kernel_size=3, stride=2, padding=1),
                      normal_layer(ngf * mult / 2), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


class LocalGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(LocalGenerator, self).__init__()
        global_model = GlobalGenerator(in_channels, out_channels).model
        global_model = [global_model[i] for i in range(len(global_model) - 3)]
        self.global_model = nn.Sequential(*global_model)

        model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
                            norm_layer(ngf), nn.ReLU(True),
                            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)]
        model_upsample = [
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf), nn.ReLU(True)]

        self.upsample_model = nn.Sequential(*model_upsample)
        self.downsample_model = nn.Sequential(*model_downsample)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        downsampled_input = self.pool(x)
        global_output = self.global_model(downsampled_input)

        output = self.upsample_model(self.downsample_model(x) + global_output)
        return output


# PatchGAN:It maps from 256*256 to an NxN outputs X,where each X_ij means whether the patch ij in the image is real.In
# the CycleGAN the receptive fields of the D turn out to be 70x70 patches in the input.
class PatchGAN(nn.Module):
    def __init__(self, in_channels, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(PatchGAN, self).__init__()
        self.getIntermFeat = getIntermFeat

        kw = 4
        padw = 2
        layers = [[nn.Conv2d(in_channels=in_channels, out_channels=ndf, kernel_size=kw, padding=padw, stride=2),
                   nn.LeakyReLU(0.2, True)]]

        layers += [[nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=kw, padding=padw, stride=2),
                    norm_layer(ndf * 2), nn.LeakyReLU(0.2, True)]]

        layers += [[nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=kw, padding=padw, stride=2),
                    norm_layer(ndf * 4), nn.LeakyReLU(0.2, True)]]

        layers += [[nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=kw, padding=padw, stride=1),
                    norm_layer(ndf * 8), nn.LeakyReLU(0.2, True)]]

        layers += [[nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=kw, padding=padw, stride=1)]]

        if use_sigmoid:
            layers += [[nn.Sigmoid()]]
        if self.getIntermFeat:
            for i in range(len(layers)):
                setattr(self, 'layer' + str(i), nn.Sequential(*layers[i]))
        else:
            model = []
            for i in range(len(layers)):
                model += layers[i]
            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.getIntermFeat:
            res = [x]
            for i in range(5):
                model = getattr(self, 'layer' + str(i))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_channel, ndf=64, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D

        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = PatchGAN(input_channel, ndf, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(5):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.pool = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singel_D_forward(self, model, x):
        if self.getIntermFeat:
            result = [x]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[-1]
        else:
            return [model(-1)]

    def forward(self, x):
        num_D = self.num_D
        result = []
        input_downsampled = x
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(5)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.pool(input_downsampled)
        return result
