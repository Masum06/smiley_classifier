# pip install Augmentor
# https://github.com/mdbloice/Augmentor
# this code distors an image from source directory
import Augmentor
i = 9
name = 0
#for i in range(10):
p = Augmentor.Pipeline(source_directory="./{}".format(name), output_directory="./data/{}".format(name), save_format="png")
p.rotate(probability=0.7, max_left_rotation=8, max_right_rotation=8)
p.zoom(probability=0.5, min_factor=1, max_factor=1.1)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=2)
p.shear(probability=.5, max_shear_left=8, max_shear_right=8) # MAX 25
# threshold
p.resize(probability=1.0, width=28, height=28)
p.sample(1000)