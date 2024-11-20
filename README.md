
# A novel recommender system using graph convolutional matrix completion and personalized knowledge-aware attention sub-network

 
This is our implementation for the following paper:

>Rasoul Hassanzadeh, Vahid Majidneghad, Bahman Arasteh


Corresponding Author: Vahid Majidneghad (vahidmn at yahoo.com)


## Introduction
In recent years, graph neural networks (GNNs) have become increasingly popular within recommender systems (RS) due to their capacity to capture essential features and interpret complex relationships. However, GNNs often struggle to fully represent fine-grained semantics within knowledge graphs (KGs) and may fall short in precisely modeling user-item interactions. To address these challenges, this paper introduces a novel approach to personalized, knowledge-aware recommendation, termed KGCMC. This model combines a user-item interaction graph with a knowledge graph to enhance the representation of interactions, facilitating the generation of more informative and contextually aware node embeddings. A central feature of KGCMC is the incorporation of Graph Convolutional Matrix Completion (GCMC), which effectively learns user and item embeddings by leveraging both user-item interaction data and the structural information in the graph. Additionally, an efficient attention-based sub-network is proposed to encode enriched semantic details from the knowledge graph into detailed item embeddings, tailored to individual users. Extensive evaluations on three benchmark datasets using F1-score and recall metrics demonstrate the proposed method’s superior performance over existing state-of-the-art approaches. The results underscore the effectiveness of integrating GCMC within personalized, knowledge-driven recommendation models, showing it can successfully mitigate current limitations and improve recommendation precision and quality.
## Environment Requirement
The code has been tested running under Python 3.9.12. The required packages are as follows:
* python == 3.9.12
* tensorflow == 2.9.1
* numpy == 1.21.5
* scipy == 1.7.3
* sklearn == 1.0.2

## Examples to Run the code
The instruction of commands has been clearly stated in the code (see src/main.py).

* Movie
```
               
python main.py  --dataset movie --aggregator concat --n_epochs 10 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type gcmc --model_type KGCN_GCMC --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```

* book
```
python main.py --dataset book --aggregator concat --n_epochs 20 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 5e-5 --layer_size [64] --adj_type norm --alg_type gcmc --model_type KGCN_GCMC --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0

```

* Music
```
python main.py --dataset music --aggregator concat --n_epochs 10 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type gcmc --model_type KGCN_GCMC --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```


```

## About implementation

We build our model based on the implementations of Personalized knowledge-aware recommendation with collaborative
and attentive graph convolutional networks (https://github.com/daiquanyu/COAT).

## About Datasets
The datasets available from the corresponding author


## Citation 
If you would like to use our code, please cite:
```

```
