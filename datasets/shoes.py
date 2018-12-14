from torchvision.datasets import ImageFolder


class Shoes(ImageFolder):
    num_classes = 21

    def __init__(self, data_dir, data_transformer):
        super(Shoes, self).__init__(data_dir, data_transformer)

