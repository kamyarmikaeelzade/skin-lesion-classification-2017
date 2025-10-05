install:
\tpip install -r requirements.txt
train:
\tpython -m melanoma_isic.train configs/vgg16.yaml
eval:
\tpython -m melanoma_isic.evaluate configs/vgg16.yaml results/models/vgg16_best.h5
