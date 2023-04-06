from ghostNet import *
from data import *
from model import UNet

model=unet(input_size=(128, 128, 3))
model.load_weights('./model/unet_membrane.hdf5', by_name=False)
testGene = testGenerator('F:/PycharmProjects/ghost_unet/data/val/image/')
results = model.predict_generator(testGene,102,verbose=1)
saveResult('F:/PycharmProjects/ghost_unet/data/val/pre/',results)