import numpy as np
import logging
import csv
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import os
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from scipy.stats import pearsonr, spearmanr
import optuna
from optuna import TrialPruned
import torch
from optuna.trial import TrialState
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiStepLR
from typing import List, Tuple, Dict, Any
import dgl
import pickle
import torch.nn.functional as F
import random
from model import EnsembleModel,orientation


folder_name = '/home/user/jiang/1/HAGGNN/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(folder_name, "combin-train_log.txt"), mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class WaterShellDataNotFoundError(Exception):
    pass


class GraphDataset(Dataset):
    def __init__(self, graph_file_paths: List[str], labels: List[float], protein_data: dict):
        self.graph_file_paths = graph_file_paths
        self.labels = labels
        self.protein_data = protein_data

    def __len__(self) -> int:
        return len(self.graph_file_paths)

    def __getitem__(self, idx: int) -> Tuple:
        graph_path = self.graph_file_paths[idx]
        graphs, _ = dgl.load_graphs(graph_path)
        graph = graphs[0]
        protein_id = extract_protein_id(self.graph_file_paths[idx].split('/')[-1])
        protein_id = protein_id.replace('.bin', '')
        num_nodes = graph.num_nodes()
        node_s = graph.ndata['s']
        src, dst = graph.edges()
        edge_index = (src.long(), dst.long())
        edge_attr = graph.edata['s']

        coords = graph.ndata['coords']
        water_shell_layers = self.get_water_shell_layers(protein_id, num_nodes)
        seq = torch.arange(num_nodes,dtype=torch.long).view(-1,1)
        ori = orientation(coords)
        return graph, node_s, coords, seq, ori, edge_index, edge_attr, self.labels[idx], water_shell_layers

    def get_water_shell_layers(self, protein_id, num_nodes):
        if protein_id in self.protein_data:
            water_shell_df = self.protein_data[protein_id]
            if isinstance(water_shell_df, list):
                water_shell_layers = [value for _, value in water_shell_df]
            else:
                water_shell_layers = water_shell_df['water_shell'].tolist()
            if len(water_shell_layers) != num_nodes:
                if len(water_shell_layers) > num_nodes:
                    water_shell_layers = water_shell_layers[:num_nodes]
                else:
                    water_shell_layers.extend([0] * (num_nodes - len(water_shell_layers)))
            return water_shell_layers

        else:
            raise ValueError(f"Water shell data for protein ID {protein_id} not found.")


def collate_fn(batch):
    graphs, node_s_list, coords_list, seq_list, ori_list, edge_indices, edge_attr_list, labels, water_shell_layers_list = zip(*batch)
    batched_graph = dgl.batch(graphs).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    node_s = batched_graph.ndata['s']
    coords = batched_graph.ndata['coords']
    seq = torch.cat(seq_list,dim=0).to(device)
    ori_tensors = []
    for ori in ori_list:
        if isinstance(ori, tuple):
            ori_tensors.append(ori[0])
        else:
            ori_tensors.append(ori)
    ori = torch.cat(ori_tensors,dim=0).to(device)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    src, dst = batched_graph.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)
    water_shell_layers = []
    
    for wsl, g in zip(water_shell_layers_list, graphs):
        num_nodes = g.num_nodes()
        if len(wsl) > num_nodes:
            water_shell_layers.extend(wsl[:num_nodes])
        else:
            water_shell_layers.extend(wsl + [0] * (num_nodes - len(wsl)))

    water_shell_layers = torch.tensor(water_shell_layers, dtype=torch.long).to(device)

    return batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, labels, water_shell_layers


def extract_protein_id(protein_id):
    return protein_id.split('-')[0].split('_')[0].strip()


def read_csv_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data


def load_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        water_shell_data = pickle.load(f)

    return water_shell_data

pickle_folder = '/home/user/jiang/watershell-pd/train'

def load_water_shell_data(pickle_folder):
    protein_data = {}
    for filename in os.listdir(pickle_folder):
        if filename.endswith('.pd'):
            pickle_file_path = os.path.join(pickle_folder, filename)
            protein_id = os.path.splitext(filename)[0]
            with open(pickle_file_path, 'rb') as f:
                water_shell_data = pickle.load(f)
                protein_data[protein_id] = water_shell_data
    return protein_data

protein_data = load_water_shell_data(pickle_folder)

def extract_graph_paths_and_labels(data, folder):
    graph_paths = []
    labels = []
    for entry in data:
        protein_id = extract_protein_id(entry['Protein_ID'])
        tm_value = float(entry['Tm'])

        for graph_filename in os.listdir(folder):
            if graph_filename.endswith(".bin"):
                graph_protein_id = graph_filename.split('.')[0]
                if protein_id == graph_protein_id:
                    path = os.path.join(folder, graph_filename)
                    graph_paths.append(path)
                    labels.append(tm_value)
                    break
    return graph_paths, labels


train_csv_path = "/home/user/jiang/train_dataset1.csv"
train_folder = "/home/user/jiang/graph/train"
test_csv = "/home/user/jiang/test2-1.csv"
test_folder = "/home/user/jiang/graph/test"
train_data = read_csv_to_dict(train_csv_path)
test_data = read_csv_to_dict(test_csv_path)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []
best_model = None
model_save_dir = '/home/user/jiang/1/HAGGNN/'
os.makedirs(model_save_dir, exist_ok=True)

device = 'cuda:0'
val_metrics_all_folds = {
    "mae": [],
    "rmse": [],
    "r2": [],
    "pcc": [],
    "spearman": []
}
test_metrics_all_folds = {
    "mae": [],
    "rmse": [],
    "r2": [],
    "pcc": [],
    "spearman": []
}

best_val_metrics_per_fold = []

patience = 20
no_improvement = 0
best_model_path = None


def train:
    drop_rate = 0.3
    hacdconv_params = {
        "geometric_radii": [4],
        "sequential_kernel_size": 21,
        "kernel_channels": [24],
        "channels": [27],
        "base_width": 64.0,
        "embedding_dim": 27,
        "batch_norm": True,
        "dropout": drop_rate,
        "bias": False,
        "water_shell_data": protein_data
    }
    logging.info(f"dropout:{drop_rate}")


    hidden_nf = 128
    out_node_nf = 64
    egnn_num_layers = 3
    egnn_params = {
        'in_node_nf': 27,
        'hidden_nf': hidden_nf,
        'out_node_nf': out_node_nf,
        'in_edge_nf': 32,
        'n_layers': egnn_num_layers,
        'attention': False,
        'water_shell_layers': protein_data
    }

    logging.info(f"egnn hidden dim:{hidden_nf},egnn out :{out_node_nf},egnn_num_layers :{egnn_num_layers}")

    mlp_hidden_dims = [128, 160, 64]
    num_epochs = 200
    random_state = 52
    lr = 0.00035000000000000005
    best_val_metric = float('-inf')
    fold_results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    batch_size = 64
    pool_method = "mean+max"
    egnn_pool_hidden_dim = 64
    egnn_pool_dropout = 0.25
    weight_decay = 5.690633755633165e-06
    grad_clip = 0.5389804449115099

    
             
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        logging.info(f"Training fold {fold + 1}/5")

        train_data_fold = [train_data[i] for i in train_idx]
        val_data_fold = [train_data[i] for i in val_idx]
        train_graph_paths, train_labels = extract_graph_paths_and_labels(train_data_fold, train_folder)
        val_graph_paths, val_labels = extract_graph_paths_and_labels(val_data_fold, train_folder)
        train_dataset = GraphDataset(train_graph_paths, train_labels, protein_data)
        val_dataset = GraphDataset(val_graph_paths, val_labels, protein_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

        model = EnsembleModel(
        hacdconv_params=hacdconv_params,
        egnn_params=egnn_params,
        mlp_hidden_dims=mlp_hidden_dims,
        output_dim=1,
        act_fn = nn.LeakyReLU(negative_slope=0.1),
        pool_method=pool_method,
        egnn_pool_hidden_dim=egnn_pool_hidden_dim,
        egnn_pool_dropout=egnn_pool_dropout
        ).to(device)


        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.3,
                patience=10,
                min_lr = 1e-6
            )

        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.L1Loss()

        best_fold_metric = float('-inf')
        best_r2_per_fold = {} 
        no_improvement = 0
        patience = 20
        fold_model_save_path = os.path.join(folder_name, f'best_model_fold_{fold}_trial_{trial.number}.pt')
        accumulation_steps = 2

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0
            accum_count = 0
            optimizer.zero_grad()

            for batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, labels, water_shell_layers in train_loader:
                node_s = node_s.to(device)
                coords = coords.to(device)
                seq = seq.to(device)
                ori = ori.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                labels = labels.clone().detach().float().to(device)
                optimizer.zero_grad()
                predictions = model(batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, water_shell_layers)
                loss = 0.7 * criterion1(predictions.squeeze().float(), labels.float()) + 0.3 * criterion2(predictions.squeeze().float(), labels.float())
                loss_scaled = loss/accumulation_steps
                loss_scaled.backward()

                epoch_train_loss += loss.item()
                accum_count += 1
                if accum_count % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

            if accum_count != 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.empty_cache()
            avg_train_loss = epoch_train_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            model.eval()
            val_predictions_list = []
            val_labels_list = []
            with torch.no_grad():
                for batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, labels, water_shell_layers in val_loader:

                    node_s = node_s.to(device)
                    coords = coords.to(device)
                    seq = seq.to(device)
                    ori = ori.to(device)
                    edge_index = edge_index.to(device)
                    edge_attr = edge_attr.to(device)
                    labels = labels.clone().detach().float().to(device)

                    val_predictions = model(batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, water_shell_layers)
                    val_predictions_list.append(val_predictions.cpu().numpy())
                    val_labels_list.append(labels.cpu().numpy())

            if len(val_predictions_list) == 0:
                val_predictions = np.array([])
            else:
                val_predictions = np.concatenate(val_predictions_list, axis=0)

            if len(val_labels_list) == 0:
                val_labels = np.array([])
            else:
                val_labels = np.concatenate(val_labels_list, axis=0)

            epsilon = 1e-10
            val_predictions = val_predictions.flatten()
            val_labels = val_labels.flatten()

            val_mae = mean_absolute_error(val_labels, val_predictions)
            val_rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))
            val_r2 = r2_score(val_labels, val_predictions)
            val_labels_adjusted = val_labels + epsilon
            val_predictions_adjusted = val_predictions + epsilon
            val_pcc, _ = pearsonr(val_labels_adjusted, val_predictions_adjusted)
            val_spearman, _ = spearmanr(val_labels_adjusted, val_predictions_adjusted)

            logging.info(
                f"Validation - fold:{fold + 1}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, PCC: {val_pcc:.4f}, Spearman:{val_spearman:.4f}")

            val_combined_metric = - (val_mae * 0.1) - (val_rmse * 0.1) + (3.5* val_r2) + (2.5 * val_pcc) + (1.5 * val_spearman)
            scheduler.step(val_combined_metric)

            if val_combined_metric > best_fold_metric:
                best_fold_metric = val_combined_metric
                torch.save(model.state_dict(), fold_model_save_path)
                logging.info(f"Best model for fold {fold} saved at epoch {epoch}")
                val_metrics_logger.info(f"New best model for fold {fold + 1} saved at epoch {epoch + 1}")
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= patience:
                val_metrics_logger.info(f"Early stopping at epoch {epoch + 1}. No improvement.")
                break

        fold_results.append(best_fold_metric)

        avg_val_metric = np.mean(fold_results)
        logging.info(f"Average best fold metric: {avg_val_metric:.4f}")

        if avg_val_metric > best_val_metric:
            best_val_metric = avg_val_metric
            best_model_path = fold_model_save_path

    return avg_val_metric


def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, labels, water_shell_layers in test_loader:
            node_s = node_s.to(device)
            coords = coords.to(device)
            eseq = seq.to(device)
            ori = ori.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            labels = labels.clone().detach().float().to(device)
            predictions = model(batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, water_shell_layers)
            preds.append(predictions.detach().cpu())
            truths.append(labels.cpu())

    pred_tensor = torch.cat(preds).float().squeeze()
    truth_tensor = torch.cat(truths).float().squeeze()
    has_nan_pred = torch.isnan(pred_tensor).any().item()
    has_inf_pred = torch.isinf(pred_tensor).any().item()
    has_nan_truth = torch.isnan(truth_tensor).any().item()
    has_inf_truth = torch.isinf(truth_tensor).any().item()

    if has_nan_pred or has_inf_pred or has_nan_truth or has_inf_truth:
        valid_mask = ~(torch.isnan(pred_tensor) | torch.isinf(pred_tensor) |
                       torch.isnan(truth_tensor) | torch.isinf(truth_tensor))
        pred_tensor = pred_tensor[valid_mask]
        truth_tensor = truth_tensor[valid_mask]
        if len(pred_tensor) == 0:
            return 0.0, 0.0, -1.0, 0.0, 0.0

    mae = mean_absolute_error(truth_tensor, pred_tensor)
    rmse = np.sqrt(mean_squared_error(truth_tensor, pred_tensor))
    r2 = r2_score(truth_tensor, pred_tensor)
    pcc, _ = pearsonr(truth_tensor.numpy(), pred_tensor.numpy())
    spearman, _ = spearmanr(truth_tensor.numpy(), pred_tensor.numpy())
    logging.info(f"Test Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, PCC={pcc:.4f}, Spearman={spearman:.4f}")

    return mae, rmse, r2, pcc, spearman


def main():
    logging.info("Loading test data...")
    graph_paths, labels, protein_data = load_data()
    test_dataset = GraphDataset(graph_paths, labels, protein_data)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    batch = next(iter(test_loader))
    hacdconv_params = {
        "geometric_radii": [4],
        "sequential_kernel_size": 21,
        "kernel_channels": [24],
        "channels": [27],
        "base_width": 64.0,
        "embedding_dim": 27,
        "batch_norm": True,
        "dropout": 0.3,
        "bias": False,
        "water_shell_data": protein_data
    }
    egnn_params = {
        'in_node_nf': 27,
        'hidden_nf': 128,
        'out_node_nf': 64,
        'in_edge_nf': 32,
        'n_layers': 3,
        'attention': False,
        'water_shell_layers': protein_data
    }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model_path = '/home/user/jiang/1/HAGGNN/best_model_fold.pt'
    logging.info(f"Evaluating model: {model_path}")

    try:
        model = EnsembleModel(
            hacdconv_params=hacdconv_params,
            egnn_params=egnn_params,
            mlp_hidden_dims=[128, 160, 64],
            output_dim=1,
            act_fn=nn.LeakyReLU(negative_slope=0.1),
            pool_method="mean+max",
            egnn_pool_hidden_dim=64,
            egnn_pool_dropout=0.25
        ).to(device)

        model.load_state_dict(torch.load(model_path))
        logging.info(f"Model loaded successfully from {model_path}")

        logging.info("\nEvaluating on Test Set:")
        mae, rmse, r2, pcc, spearman = evaluate_on_test_set(model, test_loader, device)

        logging.info(f"\nFinal Evaluation Results:")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"R2: {r2:.4f}")
        logging.info(f"PCC: {pcc:.4f}")
        logging.info(f"Spearman: {spearman:.4f}")

    except Exception as e:
        logging.error(f"Failed to evaluate model: {str(e)}")


if __name__ == "__main__":
    main()

