import torch
import torch.nn as nn
import torch.optim as optim
import torch
import os
import csv
import time
from utils.lossmeter import AverageMeter
from compressai.zoo import bmshj2018_factorized,cheng2020_anchor
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from ConfigSpace import ConfigurationSpace, Integer,Float
from smac import HyperparameterOptimizationFacade, Scenario
from smac.multi_objective.parego import ParEGO
from ConfigSpace import Configuration
from smac.initial_design.sobol_design import SobolInitialDesign

class CompressionModel:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        batch_size = Integer("batch_size", (4, 64), default=8)
        learning_rate = Float("learning_rate", (0.00005, 0.00015), default=0.0001, log=True)
        test_batch_size = Integer("test_batch_size", (4, 64), default=16)
        num_blocks = Integer("num_blocks", (2, 8), default=4)  # 假设深度在4到8之间
        cs.add_hyperparameters([batch_size, learning_rate, test_batch_size, num_blocks])
        return cs

    def __init__(self):
        # 设置超参数
        self.model_name = "custom-cheng2020-anchor"
        self.dataset_path = "/content/tiny-imagenet-200-3k"
        self.patience = 2
        self.trigger_times = 0
        self.best_loss = float('inf')
        self.epochs = 20
        self.patch_size = (256, 256)
        self.cuda = torch.cuda.is_available()
        self.save = True
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # 设备配置
        self.device = "cuda" if self.cuda else "cpu"

        # 加载模型
        # self.model = cheng2020_anchor(quality=3, pretrained=False).to(self.device)

        # 数据加载
        #self.train_dataloader, self.test_dataloader = self.load_data()

        # 配置优化器和损失函数
        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        #self.criterion = RateDistortionLoss(lmbda=self.lambda_rd)\
        #self.lambda_rd = 0.01
        #self.batch_size = config["batch_size"]
        #self.test_batch_size = 16

    def log_to_csv(self, log_data, config_id):
        log_file = os.path.join(self.log_dir, f"log_{config_id}.csv")
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'batch_index', 'train_loss', 'test_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # 文件不存在则写入表头

            for data in log_data:
                writer.writerow(data)

    def load_data(self):
        # 定义数据预处理
        train_transforms = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
        ])

        # 加载数据集
        train_dataset = ImageFolder(root=f"{self.dataset_path}", transform=train_transforms)
        test_dataset = ImageFolder(root=f"{self.dataset_path}", transform=test_transforms)

        # 数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.cuda,
            num_workers=4
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.cuda,
            num_workers=4
        )
        return train_dataloader, test_dataloader

    def test_epoch(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        bpp_loss_meter = AverageMeter()
        mse_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        with torch.no_grad():
            for data in self.test_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                losses = self.criterion(output, data)

                loss_meter.update(losses['loss'].item())
                bpp_loss_meter.update(losses['bpp_loss'].item())
                mse_loss_meter.update(losses['mse_loss'].item())
                aux_loss_meter.update(self.model.aux_loss().item())

        print(f"Test epoch {epoch}: Average losses:"
              f"\tLoss: {loss_meter.avg:.3f} |"
              f"\tMSE loss: {mse_loss_meter.avg:.3f} |"
              f"\tBpp loss: {bpp_loss_meter.avg:.2f} |"
              f"\tAux loss: {aux_loss_meter.avg:.2f}\n")
        return loss_meter.avg

    def reset_model(self, learning_rate, num_blocks):
        num_blocks=int(num_blocks)
        print("reset_numblocks", num_blocks,learning_rate)
        self.model = CustomCheng2020Anchor(N=192, num_blocks=num_blocks).to(self.device)
        self.model.print_model_structure()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = RateDistortionLoss(lmbda=0.015)

    def train(self,config,seed=None):
        print("Training with config:", config)
        self.reset_model(config["learning_rate"], config["num_blocks"])
        self.batch_size = config["batch_size"]
        self.test_batch_size = config["test_batch_size"]
        log_data = []
        start_time = time.time()
        config_id = f"config_{config['batch_size']}_{config['learning_rate']:.5f}_{config['test_batch_size']}"
        # self.learning_rate = config["learning_rate"]
        # self.lambda_rd = config["lambda_rd"]
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = RateDistortionLoss(lmbda=self.lambda_rd)
        self.train_dataloader, self.test_dataloader = self.load_data()
        best_loss = float('inf')
        total_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            self.model.train()
            for data in self.train_dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data)['loss']
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                log_data.append({
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'test_loss': None  # 测试损失稍后填写
                })
                print(f" Loss: {loss.item()}")

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            total_loss += avg_epoch_loss
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            # 测试阶段
            test_loss = self.test_epoch(epoch + 1)
            for item in log_data:
                if item['epoch'] == epoch + 1:
                    item['test_loss'] = test_loss

            is_best = test_loss < best_loss
            if is_best:
                self.best_loss = test_loss
                if self.save:
                    torch.save(self.model.state_dict(), f"{self.model_name}_best_di.pth")
            if not is_best:
                self.trigger_times += 1
                if self.trigger_times >= self.patience:
                    print("Early stopping triggered due to no improvement!")
                    break
            else:
                self.trigger_times = 0  # Reset the trigger times if there is improvement

            average_loss = total_loss / self.epochs
        self.log_to_csv(log_data, config_id)
        return {
            "loss": average_loss
            #"time": time.time() - start_time,
        }
        return {}

if __name__ == "__main__":
    compressor = CompressionModel()
    objectives = ["loss"]
    scenario = Scenario(
        compressor.configspace,
        objectives=objectives,
        n_trials=40,
        n_workers=1,
    )
    initial_config = Configuration(
        compressor.configspace,
        values={
            'batch_size': 8,
            'learning_rate': 0.0001,
            'num_blocks': int(5),
            'test_batch_size': 8,
        }
    )

    initial_design = SobolInitialDesign(
        scenario=scenario,
        additional_configs=[initial_config],
        n_configs=0  # 确保只有一个初始配置
    )
    #initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)
    #multi_objective_algorithm = ParEGO(scenario)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)

    smac = HyperparameterOptimizationFacade(
        scenario,
        compressor.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
    )

    incumbent = smac.optimize()

    print("Configuration", incumbent[0])

