import cv2

# path = "FLIC/images/2-fast-2-furious-00019871.jpg"
# img = cv2.imread(path)

# print(path)
# cv2.imshow(path, img)
# cv2.waitKey(0)

import torch
x = torch.rand(5, 3)
print(x)

    # def __init__(self):
    #     super(KeypointModel, self).__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),

    #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),

    #         nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.AdaptiveAvgPool2d((1, 1))
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(128, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, num_outputs)
    #     )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.classifier(x)
    #     return x