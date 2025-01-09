

import os
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models import BadNet
from utils import print_model_perform
import copy
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum=0, weight_decay=0):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def backdoor_model_trainer(dataname, poisoned_portion, train_data_loader, test_data_ori_loader, test_data_tri_loader, trigger_label,
                           epoch, batch_size, loss_mode, optimization, lr, print_perform_every_epoch, basic_model_path,
                           device):
    badnet = BadNet(input_channels=train_data_loader.dataset.channels,
                    output_num=train_data_loader.dataset.class_num).to(device)
    criterion = loss_picker(loss_mode)
    optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)

    train_process = []
    print("### target label is %d, EPOCH is %d, Learning Rate is %f" % (trigger_label, epoch, lr))
    print("### Train set size is %d, ori test set size is %d, tri test set size is %d\n" % (
    len(train_data_loader.dataset), len(test_data_ori_loader.dataset), len(test_data_tri_loader.dataset)))
    for epo in range(epoch):
        loss = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
        acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor',
                         print_perform=print_perform_every_epoch)
        acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor',
                            print_perform=print_perform_every_epoch)
        acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor',
                            print_perform=print_perform_every_epoch)

        print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n" \
              % (epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))

        # save model
        torch.save(badnet.state_dict(), basic_model_path)

        # save training progress
        train_process.append(
            (dataname, batch_size, trigger_label, lr, epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        df = pd.DataFrame(train_process, columns=(
        "dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc",
        "test_tri_acc"))
        df.to_csv("./logs/%s_train_process_trigger%d_poisoned%.1f.csv" % (dataname, trigger_label, poisoned_portion), index=False, encoding='utf-8')

    return badnet


def train(model, data_loader, criterion, optimizer, loss_mode, device='cpu'):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate((data_loader)):
        # print(batch_y.size())
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)  # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y)  # mse loss
        elif loss_mode == "cross":
            # print(batch_y)
            # print(output)
            loss = criterion(output, batch_y)#torch.argmax(batch_y, dim=1))  # cross entropy loss
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu'):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        # if mode == 'pruned':
        #     batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()




    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


def eval_dif(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu'):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        # if mode == 'pruned':
        #     batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc




###############################
# trainer for Membership inference attack
###############################

# Prepare data for Attack Model
def prepare_attack_data(model,
                        iterator,
                        device,
                        top_k=False,
                        test_dataset=False):
    attackX = []
    attackY = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, label in iterator:
            labels.append(label)

            # Move tensors to the configured device
            inputs = inputs.to(device)

            # Forward pass through the model
            outputs = model(inputs)

            # To get class probabilities
            posteriors = F.softmax(outputs, dim=1)
            if top_k:
                # Top 3 posterior probabilities(high to low) for train samples
                topk_probs, _ = torch.topk(posteriors, 3, dim=1)
                attackX.append(topk_probs.cpu())
            else:
                attackX.append(posteriors.cpu())

            # This function was initially designed to calculate posterior for training loader,
            # but to handle the scenario when trained model is given to us, we added this boolean
            # to different if the dataset passed is training or test and assign labels accordingly
            if test_dataset:
                attackY.append(torch.zeros(posteriors.size(0), dtype=torch.long))
            else:
                attackY.append(torch.ones(posteriors.size(0), dtype=torch.long))

    return attackX, attackY, labels


def train_per_epoch(model,
                    train_iterator,
                    criterion,
                    optimizer,
                    device,
                    bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    model.train()
    for _, (features, target) in enumerate(train_iterator):
        # Move tensors to the configured device
        features = features.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(features)
        if bce_loss:
            # For BCE loss
            loss = criterion(outputs, target.unsqueeze(1))
        else:
            loss = criterion(outputs, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record Loss
        epoch_loss += loss.item()

        # Get predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Per epoch valdication accuracy calculation
    epoch_acc = correct / total
    epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc


def val_per_epoch(model,
                  val_iterator,
                  criterion,
                  device,
                  bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _, (features, target) in enumerate(val_iterator):
            features = features.to(device)
            target = target.to(device)

            outputs = model(features)
            # Caluclate the loss
            if bce_loss:
                # For BCE loss
                loss = criterion(outputs, target.unsqueeze(1))
            else:
                loss = criterion(outputs, target)

            # record the loss
            epoch_loss += loss.item()

            # Check Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Per epoch valdication accuracy and loss calculation
        epoch_acc = correct / total
        epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc


###############################
# Training Attack Model
###############################
def train_attack_model(model,
                       dataset,
                       criterion,
                       optimizer,
                       lr_scheduler,
                       device,
                       model_path='./model',
                       epochs=10,
                       b_size=20,
                       num_workers=1,
                       verbose=False,
                       earlystopping=False):
    n_validation = 1000  # number of validation samples
    best_valacc = 0
    stop_count = 0
    patience = 5  # Early stopping

    path = os.path.join(model_path, 'best_attack_model.ckpt')

    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []

    train_X, train_Y = dataset

    # Contacetnae list of tensors to a single tensor
    t_X = torch.cat(train_X)
    t_Y = torch.cat(train_Y)

    # #Create Attack Dataset
    attackdataset = TensorDataset(t_X, t_Y)

    print('Shape of Attack Feature Data : {}'.format(t_X.shape))
    print('Shape of Attack Target Data : {}'.format(t_Y.shape))
    print('Length of Attack Model train dataset : [{}]'.format(len(attackdataset)))
    print('Epochs [{}] and Batch size [{}] for Attack Model training'.format(epochs, b_size))

    # Create Train and Validation Split
    n_train_samples = len(attackdataset) - n_validation
    train_data, val_data = torch.utils.data.random_split(attackdataset,
                                                         [n_train_samples, n_validation])

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=b_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=b_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    print('----Attack Model Training------')
    for i in range(epochs):

        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        lr_scheduler.step()

        print('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
              .format(i + 1, epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

        if earlystopping:
            if best_valacc <= valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= patience:  # early stopping check
                    print('End Training after [{}] Epochs'.format(epochs + 1))
                    break
        else:  # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, path)

    return best_valacc


###################################
# Training Target and Shadow Model
###################################
def train_model(model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                optimizer,
                scheduler,
                device,
                model_path,
                verbose=False,
                num_epochs=50,
                top_k=False,
                earlystopping=False,
                is_target=False):
    best_valacc = 0
    patience = 5  # Early stopping
    stop_count = 0
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []

    if is_target:
        print('----Target model training----')
    else:
        print('---Shadow model training----')

    # Path for saving best target and shadow models
    target_path = os.path.join(model_path, 'best_target_model.ckpt')
    shadow_path = os.path.join(model_path, 'best_shadow_model.ckpt')

    for epoch in range(num_epochs):

        train_loss, train_acc = train_per_epoch(model, train_loader, loss, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, loss, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        scheduler.step()

        print('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
              .format(epoch + 1, num_epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

        if earlystopping:
            if best_valacc <= valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                if is_target:
                    torch.save(best_model, target_path)
                else:
                    torch.save(best_model, shadow_path)
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= patience:  # early stopping check
                    print('End Training after [{}] Epochs'.format(epoch + 1))
                    break
        else:  # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            if is_target:
                torch.save(best_model, target_path)
            else:
                torch.save(best_model, shadow_path)

    if is_target:
        print('----Target model training finished----')
        print('Validation Accuracy for the Target Model is: {:.2f} %'.format(100 * best_valacc))
    else:
        print('----Shadow model training finished-----')
        print('Validation Accuracy for the Shadow Model is: {:.2f} %'.format(100 * best_valacc))

    if is_target:
        print('----LOADING the best Target model for Test----')
        model.load_state_dict(torch.load(target_path))
    else:
        print('----LOADING the best Shadow model for Test----')
        model.load_state_dict(torch.load(shadow_path))

    # As the model is fully trained, time to prepare data for attack model.
    # Training Data for members would come from shadow train dataset, and member inference from target train dataset respectively.
    attack_X, attack_Y, _ = prepare_attack_data(model, train_loader, device, top_k)

    # In test phase, we don't need to compute gradients (for memory efficiency)
    print('----Test the Trained Network----')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            test_outputs = model(inputs)

            # Predictions for accuracy calculations
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Posterior and labels for non-members
            probs_test = F.softmax(test_outputs, dim=1)
            if top_k:
                # Take top K posteriors ranked high ---> low
                topk_t_probs, _ = torch.topk(probs_test, 3, dim=1)
                attack_X.append(topk_t_probs.cpu())
            else:
                attack_X.append(probs_test.cpu())
            attack_Y.append(torch.zeros(probs_test.size(0), dtype=torch.long))

        if is_target:
            print('Test Accuracy of the Target model: {:.2f}%'.format(100 * correct / total))
        else:
            print('Test Accuracy of the Shadow model: {:.2f}%'.format(100 * correct / total))

    return attack_X, attack_Y