from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from utils import SPLM, accuracy, bootstrap_latent


TRAIN_DATASET = STL10(root="data", split="train", download=True, transform=ToTensor())
TRAIN_UNLABELED_DATASET = STL10(
    root="data", split="train+unlabeled", download=True, transform=ToTensor()
)
TEST_DATASET = STL10(root="data", split="test", download=True, transform=ToTensor())

model = resnet18(pretrained=True)
supervised = SPLM(model)
trainer = pl.Trainer(max_epochs=25, gpus=-1)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=128,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    TEST_DATASET,
    batch_size=128,
)
trainer.fit(supervised, train_loader, val_loader)

model.cuda()
acc = sum([accuracy(model(x.cuda()), y.cuda()) for x, y in val_loader]) / len(val_loader)
print(f"Accuracy: {acc:.3f}")

#Self-supervised training using bootstrapping method

model = resnet18(pretrained=True)
byol = bootstrap_latent(model, image_size=(96, 96))
trainer = pl.Trainer(
    max_epochs=50, 
    gpus=-1,
    accumulate_grad_batches=2048 // 128
)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=128,
    shuffle=True,
    drop_last=True,
)
trainer.fit(byol, train_loader, val_loader)

#Resume supervised training
state_dict = model.state_dict()
model = resnet18()
model.load_state_dict(state_dict)

supervised = SPLM(model)
trainer = pl.Trainer(
    max_epochs=25, 
    gpus=-1
)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=128,
    shuffle=True,
    drop_last=True,
)
trainer.fit(supervised, train_loader, val_loader)

model.cuda()
acc = sum([accuracy(model(x.cuda()), y.cuda()) for x, y in val_loader]) / len(val_loader)
print(f"Accuracy: {acc:.3f}")
