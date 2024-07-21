from bread.algo.tracking import AssignmentDataset
from skorch import NeuralNetClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np, scipy.optimize, scipy.sparse
from typing import List, Union
import torch

__all__ = ['AssignmentClassifier', 'GraphLoader','seed_torch']

def seed_torch(seed=42):
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# Function to print the device being used
def print_device():
    if torch.cuda.is_available():
        print("Running on GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU")

class AssignmentClassifier(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device  # Set the device for the model
    def predict_assignment(self, data: Union[AssignmentDataset, Data], assignment_method: str = 'default', return_dict: bool = False, return_middle_values: bool = False) -> Union[List[np.ndarray], np.ndarray]:
        if not isinstance(data, Data):
            return [self.predict_assignment(graph) for graph in data ]

        graph = data
        cell_ids1 = set([ i for i, a in list(graph.cell_ids) ])
        cell_ids2 = set([ a for i, a in list(graph.cell_ids)])
        n1, n2 = len(cell_ids1), len(cell_ids2)
        # z,middle_layer = self.evaluation_step((graph, None), training=False,)  # pass return_middle_values argument
        z = self.evaluation_step((graph, None), training=False,)
        z = z.cpu().numpy() 
        scores = z.reshape((n1, n2))
        if(assignment_method == 'default' or assignment_method == 'hungarian'):
            assignment = self.hungarian(scores,n1,n2)
        elif(assignment_method == 'modified_hungarian'):
            assignment = self.modified_hungarian(scores,n1,n2)
        elif(assignment_method == 'Jonker_Volgenant'):
            assignment = self.Jonker_Volgenant(scores,n1,n2)
        elif(assignment_method == 'squared_hungarian'):
            assignment = self.squared_hungarian(scores,n1,n2)
        elif(assignment_method == 'modified_hungarian2'):
            assignment = self.modified_hungarian2(scores,n1,n2)
        elif(assignment_method == 'percentage_hungarian'):
            assignment = self.percentage_hungarian(scores,n1,n2, data = graph)
        elif(assignment_method == 'custom_optimizer'):
            assignment = self.custom_optimizer(scores,n1,n2, data = graph)
        else:
            raise ValueError(f'assignment_method {assignment_method} not recognized')
        if return_dict:
            dict_frame1 = dict(enumerate(cell_ids1))
            dict_frame2 = dict(enumerate(cell_ids2))
            # make a dictionary of cell_id1 to cell_id2 based on the assignment matrix
            # assignment_dict = { cell_to_id1[j]: cell_to_id2[i] for i, j in zip(*np.where(assignment)) }
            assignment_dict = {}

            for i in range(len(assignment)):
                for j in range(len(assignment[i])):
                    if assignment[i][j] == 1:
                        cell_id_frame1 = dict_frame1[i]
                        cell_id_frame2 = dict_frame2[j]
                        assignment_dict[cell_id_frame2] = cell_id_frame1

            # Assign -1 to cells in the second frame that were not assigned to any cell in the first frame
            for cell_id_frame2 in dict_frame2.values():
                if cell_id_frame2 not in assignment_dict:
                    assignment_dict[cell_id_frame2] = -1

            return assignment_dict
            
        else:
            if return_middle_values:
                return assignment, middle_layer
            else:
                return assignment
    
    def predict_raw(self, data: Union[AssignmentDataset, Data]) -> Union[List[np.ndarray], np.ndarray]:
        if not isinstance(data, Data):
            return [self.predict_assignment(graph) for graph in data ]

        graph = data
        cell_ids1 = set([ i for i, a in list(graph.cell_ids) ])
        cell_ids2 = set([ a for i, a in list(graph.cell_ids)])
        n1, n2 = len(cell_ids1), len(cell_ids2)
        z = self.evaluation_step((graph, None), training=False).cpu().numpy()  # perform raw forward pass
        scores = z.reshape((n1, n2))
        return scores

    def percentage_hungarian(self, scores: np.ndarray, n1: int ,n2: int , data=None) -> np.ndarray:
        # Create a copy of the scores matrix
        scores_copy = scores.copy()

        # Solving the LAP using the Jonker-Volgenant algorithm
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(scores_copy, maximize=True)  # maximizing the total assignment score

        # 'row_ind' and 'col_ind' are the row and column indices of the optimal assignments.
        # Create the binary assignment matrix
        binary_assignment_matrix = np.zeros_like(scores, dtype=int)
        binary_assignment_matrix[row_ind, col_ind] = 1

        # Calculate the percentage of assignment scores that are higher for each assigned cell
        percentage_higher_scores = []
        threshold = 5  # Adjust the threshold as needed
        for i, j in zip(row_ind, col_ind):
            if binary_assignment_matrix[i, j] == 1:
                count_higher_scores_j = np.sum(scores_copy[:, j] > scores[i, j])
                count_higher_scores_i = np.sum(scores_copy[i, :] > scores[i, j])
                count_higher_scores = (count_higher_scores_j + count_higher_scores_i)/2
                percentage = (count_higher_scores / (scores.shape[0] - 1)) * 100  # excluding the current cell
                percentage_higher_scores.append(percentage)
                if percentage > threshold:
                    binary_assignment_matrix[:, j] = 0  # Set all assignments to this cell to 0
        # print(percentage_higher_scores)
        print("min_percentage_higher_scores: ", np.min(percentage_higher_scores), "max_percentage_higher_scores: ", np.max(percentage_higher_scores), "average_percentage_higher_scores: ", np.mean(percentage_higher_scores))
        return binary_assignment_matrix
    
    def hungarian(self, scores: np.ndarray, n1: int ,n2: int) -> np.ndarray:
        yx = scipy.optimize.linear_sum_assignment(scores, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        return assignment

    def modified_hungarian(self, scores: np.ndarray, n1: int ,n2: int, threshold: float = 0.01) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        scores_modified[scores_modified < threshold] = 0
        yx = scipy.optimize.linear_sum_assignment(scores_modified, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        # create a boolean mask of elements that are zero in scores_modified
        mask = scores_modified == 0
        # set the corresponding elements in assignment to zero
        assignment[mask] = 0
        return assignment

    def sigmoid_hungarian(self, scores: np.ndarray, n1: int ,n2: int,) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        yx = scipy.optimize.linear_sum_assignment(scores_modified, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        return assignment

    def squared_hungarian(self, scores: np.ndarray, n1: int ,n2: int) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        scores = scores_modified**2
        yx = scipy.optimize.linear_sum_assignment(scores, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        return assignment
    
    def custom_optimizer(self, scores: np.ndarray, n1: int ,n2: int, data = None) -> np.ndarray:
        from scipy.optimize import linprog

        similarity_matrix = 1 / (1 + np.exp(-scores))  # apply sigmoid function
        # Linear programming coefficients for the objective function (negative because linprog does minimization)
        k = n1
        if n1 < n2:
            dummy_objects = np.zeros((n2 - n1, n2))
            similarity_matrix = np.vstack([similarity_matrix, dummy_objects])
            k = 2*n1-n2


        elif n1 > n2:
            dummy_objects = np.zeros((n1, n1 - n2))
            similarity_matrix = np.hstack([similarity_matrix, dummy_objects])
            k = n1

        
        c = -similarity_matrix.flatten()
        
        n = np.max([n1,n2])
        
        # Coefficients for the equality constraint (sum of assignments should be k)
        A_eq = np.ones((1, n * n))
        b_eq = np.array([k])

        # Coefficients for the inequality constraints (at most one assignment per row and column)
        A_ub = np.vstack([np.kron(np.eye(n), np.ones((1, n))),
                        np.kron(np.ones((1, n)), np.eye(n))])
        b_ub = np.ones((2 * n,))

        # Bounds for decision variables (binary assignments)
        bounds = [(0, 1) for _ in range(n * n)]

        # Solve the linear programming problem
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,  bounds=(0, 1), method='highs')

        # Extract the solution (reshape to a matrix)
        assignment_matrix = np.abs(result.x.reshape((n, n)))
        assignment_matrix = assignment_matrix[:n1, :n2]

        return assignment_matrix


class GraphLoader(DataLoader):
    # we need this class to load the graph data into the training loop, because graphs are dynamically sized and can't be stored as normal numpy arrays
    # https://github.com/skorch-dev/skorch/blob/8db8a0d4d23e696c54cc96494b54a83f5ac55d69/notebooks/CORA-geometric.ipynb
    def __iter__(self):
        it = super().__iter__()
        for graph in it:
            yield graph, graph.y.float()