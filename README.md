# Yu-GAN-Oh
A Generative Adversarial Network

## This project uses a GAN to generate new Yu-Gi-Oh card arts based on a large library of images.

The source code for all of the project is contained in yuGANoh.py. 

The script is filled with redundant lines that were used in spyder because lines can be run one at a time easily. The process took about 1.5 hours for me on the 7000 55x55px images used, and 3 hours for 200x200px images. This was using a i5-6300 CPU and a Nvidia 960M GPU. An animation is available as the gif in the repository.

Check out the gif!

![alt text](https://github.com/kasplat/yu-GAN-oh/blob/master/YuGANohAnimated.gif)

## Process

Creating this was quite tricky. First, I needed to find some images.
I found them using the Image updater here: <https://github.com/jonas-git/image-updater>.
Next, I transformed them into a form that was usable by the generator and discriminator.
After that, I created networks for the generator and discriminator.
Then, I wrote the logic to feed one into the other and train each other.
Finally, I ran it all together for several epochs with varying hyperparameters until I found a solution that worked.

## Acknowledgments

* Yvan Scher for inspiration. He developed a similar GAN to the one described in Yvan Scher's Monster GAN project here: <https://medium.com/@yvanscher/using-gans-to-create-monsters-for-your-game-c1a3ece2f0a0>
* Dandyhack for the great hackathon and space!
