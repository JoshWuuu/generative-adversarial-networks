# CycleGAN
## Summary
It is a side project in 2023. The project is a Computer Vision topic. The languages and relevent packages are **Python - Pytorch**. The repo built the CycleGAN to train the famous zebra and horse dataset. 
## Data
The zebra to horse dataset, [source](https://www.kaggle.com/datasets/suyashdamle/cyclegan)
## Network
CycleGAN is the double DCGAN. The network has two discriminators and two generators (For this dataset, it will be horse discriminator, horse generator, zebra discriminator and zebra generator). The network forms an cycle with two generators and two discriminators. The generator is a encoder and decoder network. The encode downsample the image into the vectors and the decoder upsample the vectors back to the images with the original size.  The discriminator is a four layer convolutional layers with stride 2, the output will be 3*3 grid with 0/1 for each cell
## Loss
Loss consists of two loss, discriminator loss and generator loss.The generator loss consists of two adversarial loss and two cycle consistency loss and two identity loss. The discriminator loss is the classfication loss of real/fake images. Discrimantor $D$ wants to seperate the real($x$)/fake($z$) images as much as possible, whereas generator $G$ wants to put the real($x$)/fake($z$) images as close as possible.
$$D_{Loss} = min\Bigl(MSE(0, D(G(z))) + MSE(1, D(x))\Bigl)$$
$$G_{Loss} = min\Bigl(MSE(1, D(G(z)) + L1(x, D(D(z)) + IdentityLoss\Bigl)$$
## Result
|1st column: Input / 2nd column: Generated / 3rd row: Re-converted|
|:---:|
|![](results/horse_results.png)|

## Reference
* https://github.com/aladdinpersson/Machine-Learning-Collection
```
@misc{zhu2020unpaired,
      title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
      author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
      year={2020},
      eprint={1703.10593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
