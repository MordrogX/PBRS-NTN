import torch
import tensorflow as tf
import pytorch_lightning as pl

class TenorNetworkModule(pl.LightningModule):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, output_dim, input_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim,
                                                             self.input_dim,
                                                             self.output_dim))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.output_dim,
                                                                   2*self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.output_dim, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.matmul(torch.t(embedding_1), self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(self.input_dim, self.output_dim)
        scoring = torch.matmul(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring)
        return scores