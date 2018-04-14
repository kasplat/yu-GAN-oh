# Yu-GAN-Oh
A Generative Adversarial Network

## This project uses a GAN to generate new Yu-Gi-Oh card arts based on a large library of images.
implements a similar GAN to the one described in Yvan Scher's Monster GAN project here: https://medium.com/@yvanscher/using-gans-to-create-monsters-for-your-game-c1a3ece2f0a0. Images were taken using Image Updater: https://github.com/jonas-git/image-updater. The source code for all of the project is contained in yuGANoh.py. 

The script is filled with redundant lines that were used in spyder because lines can be run one at a time easily. The process took about 1.5 hours for me on the 7000 55x55px images used, and 3 hours for 200x200px images. This was using a i5-6300 CPU and a Nvidia 960M GPU. An animation is available as the gif in the repository, and all pictures are on root because I got tired and couldn't figure out git at 4am. 

The Net uses 5 convolutional layers for both the generator and discriminator. ReLU, leakyReLU, and tanh were used as activation layers.

Check out the gif!

![alt text](https://github.com/kasplat/yu-GAN-oh/blob/master/YuGANohAnimated.gif)

Thanks for checking out this project!
