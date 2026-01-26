# HAGGNN: Hydration-Aware Geometric Graph Neural Networks

This repository provides the implementation of **HAGGNN**, a hydration-aware geometric graph neural network framework for **protein thermal stability prediction**.  
The framework incorporates explicit hydration shell information into geometric GNNs and supports multiple backbone architectures.

---

## 📁 Repository Structure


---

## 🗂️ Folder Descriptions

### `data/`
Contains the dataset splits and corresponding protein structures.  
This folder includes **training and test CSV files** as well as their associated **PDB files**, with large structure files managed using **Git LFS**.

### `haggnn/`
Implementation of **HAGGNN**, the **main model proposed in this work**, which explicitly incorporates hydration shell information into geometric message passing.

### `ha-egnn/`
Hydration-aware extension of the Equivariant Graph Neural Network (EGNN) used as a baseline model.

### `ha-gearnet/`
Hydration-aware version of GearNet for modeling protein structures with hydration-modulated interactions.

### `ha-gvp-gnn/`
Hydration-aware adaptation of GVP-GNN incorporating hydration information into scalar and vector features.

### `ha-cdconv/`
Hydration-aware variant of CDConv with hydration-informed convolutional operations.

### `ha-gcpnet/`
Hydration-aware implementation of GCPNet integrating hydration shell features into geometric message passing.

---

## 📊 Dataset

The `data/` directory contains:
- Training and test splits (`train_dataset.csv`, `test_dataset.csv`)
- Corresponding protein structure files in **PDB format**
- Large PDB files tracked using **Git LFS**

---

## ⚙️ Requirements

The code is developed and tested under the following environment:

- **Python**: 3.9.20  
- **PyTorch**: 2.1.0  
- **DGL**: 2.4.0 + cu118  
- **PyTorch Geometric**: 2.6.1  
- **NumPy**: 1.26.4  

---


