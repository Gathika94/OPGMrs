import torch
from torch import Tensor
from itertools import combinations
from src.utils.config import cfg
from IPython.core.debugger import Tracer

import tempfile
import shutil
import itertools
from scipy.sparse import coo_matrix
from pygmtools.dataset import *
import json


def pck(x: Tensor, x_gt: Tensor, perm_mat: Tensor, dist_threshs: Tensor, ns: Tensor) -> Tensor:
    r"""
    Percentage of Correct Keypoints (PCK) evaluation metric.

    If the distance between predicted keypoint and the ground truth keypoint is smaller than a given threshold, than it
    is regraded as a correct matching.

    This is the evaluation metric used by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    :param x: :math:`(b\times n \times 2)` candidate coordinates. :math:`n`: number of nodes in input graph
    :param x_gt: :math:`(b\times n_{gt} \times 2)` ground truth coordinates. :math:`n_{gt}`: number of nodes in ground
     truth graph
    :param perm_mat: :math:`(b\times n \times n_{gt})` permutation matrix or doubly-stochastic matrix indicating
     node-to-node correspondence
    :param dist_threshs: :math:`(b\times m)` a tensor contains thresholds in pixel. :math:`m`: number of thresholds for
     each batch
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(m)` the PCK values of this batch

    .. note::
        An example of ``dist_threshs`` for 4 batches and 2 thresholds:
        ::

            [[10, 20],
             [10, 20],
             [10, 20],
             [10, 20]]
    """
    device = x.device
    batch_num = x.shape[0]
    thresh_num = dist_threshs.shape[1]

    indices = torch.argmax(perm_mat, dim=-1)

    dist = torch.zeros(batch_num, x_gt.shape[1], device=device)
    for b in range(batch_num):
        x_correspond = x[b, indices[b], :]
        dist[b, 0:ns[b]] = torch.norm(x_correspond - x_gt[b], p=2, dim=-1)[0:ns[b]]

    match_num = torch.zeros(thresh_num, device=device)
    total_num = torch.zeros(thresh_num, device=device)
    for b in range(batch_num):
        for idx in range(thresh_num):
            matches = (dist[b] < dist_threshs[b, idx])[0:ns[b]]
            match_num[idx] += torch.sum(matches).to(match_num.dtype)
            total_num[idx] += ns[b].to(total_num.dtype)

    return match_num / total_num


def matching_recall(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Recall between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching recall} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}^{gt}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching recall

    .. note::
        This function is equivalent to "matching accuracy" if the matching problem has no outliers.
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    acc = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        acc[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_gt[b, :ns[b]])

    acc[torch.isnan(acc)] = 1

    return acc


def matching_precision(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Precision between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching precision} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching precision

    .. note::
        This function is equivalent to "matching accuracy" if the matching problem has no outliers.
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    precision = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        precision[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_pred[b, :ns[b]])

    precision[torch.isnan(precision)] = 1

    return precision


def matching_recall_varied(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Recall between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching recall} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}^{gt}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b\times 2)` number of nodes in all pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching recall

    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'

    acc = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        mask = torch.zeros((pmat_pred[b].shape[0],pmat_gt[b].shape[1]),device=pmat_pred.device)
        mask[:ns[0][b]+1,:ns[1][b]+1]=1
        mask[ns[0][b], ns[1][b]]=0
        acc[b] = torch.sum(pmat_pred[b] * pmat_gt[b] * mask) / torch.sum(pmat_gt[b] * mask)
        # acc[b] = torch.sum(pmat_pred[b, :ns[0][b], :ns[1][b]] * pmat_gt[b, :ns[0][b], :ns[1][b]]) / torch.sum(
        #     pmat_gt[b, :ns[0][b], :ns[1][b]])

    acc[torch.isnan(acc)] = 0

    return acc


def matching_precision_varied(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Precision between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching precision} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b\times 2)` number of nodes in all pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching precision

    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'

    precision = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        mask = torch.zeros((pmat_pred[b].shape[0],pmat_gt[b].shape[1]),device=pmat_pred.device)
        mask[:ns[0][b]+1,:ns[1][b]+1]=1
        mask[ns[0][b], ns[1][b]]=0
        precision[b] = torch.sum(pmat_pred[b] * pmat_gt[b] * mask) / torch.sum(pmat_pred[b] * mask)
        # precision[b] = torch.sum(pmat_pred[b, :ns[0][b]+1, :ns[1][b]+1] *
        #                          pmat_gt[b, :ns[0][b]+1, :ns[1][b]+1]) / torch.sum(pmat_pred[b, :ns[0][b]+1, :ns[1][b]+1])

    precision[torch.isnan(precision)] = 0

    return precision


def matching_accuracy(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor, idx: int) -> Tensor:
    r"""
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching recall} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}^{gt}}

    This function is a wrapper of ``matching_recall``.

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b\times g)` number of nodes in graphs, where :math:`g=2` for 2GM, and :math:`g>2` for MGM. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :param idx: :math:`(int)` index of source graph in the graph pair.

    :return: :math:`(b)` matching accuracy

    .. note::
        If the graph matching problem has no outliers, it is proper to use this metric and papers call it "matching
        accuracy". If there are outliers, it is better to use ``matching_precision`` and ``matching_recall``.
    """
    if 'gcan' in cfg.MODEL_NAME and 'afat' not in cfg.MODEL_NAME:
        return matching_recall_varied(pmat_pred, pmat_gt, ns)
    else:
        return matching_recall(pmat_pred, pmat_gt, ns[idx])


def format_accuracy_metric(ps: Tensor, rs: Tensor, f1s: Tensor) -> str:
    r"""
    Helper function for formatting precision, recall and f1 score metric

    :param ps: tensor containing precisions
    :param rs: tensor containing recalls
    :param f1s: tensor containing f1 scores
    :return: a formatted string with mean and variance of precision, recall and f1 score

    Example output:
    ::

        p = 0.7837±0.2799, r = 0.7837±0.2799, f1 = 0.7837±0.2799
    """
    return 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
        .format(torch.mean(ps), torch.std(ps), torch.mean(rs), torch.std(rs), torch.mean(f1s), torch.std(f1s))

def format_metric(ms: Tensor) -> str:
    r"""
    Helping function for formatting single metric.

    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}±{:.4f}'.format(torch.mean(ms), torch.std(ms))


def objective_score(pmat_pred: Tensor, affmtx: Tensor) -> Tensor:
    r"""
    Objective score given predicted permutation matrix and affinity matrix from the problem.

    .. math::
        \text{objective score} = \mathrm{vec}(\mathbf{X})^\top \mathbf{K} \mathrm{vec}(\mathbf{X})

    where :math:`\mathrm{vec}(\cdot)` means column-wise vectorization.

    :param pmat_pred: predicted permutation matrix :math:`(\mathbf{X})`
    :param affmtx: affinity matrix of the quadratic assignment problem :math:`(\mathbf{K})`
    :return: objective scores

    .. note::
        The most general mathematical form of graph matching is known as Quadratic Assignment Problem (QAP), which is an
        NP-hard combinatorial optimization problem. Objective score reflects the power of the graph matching/QAP solver
        concerning the objective score of the QAP.
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score

def clustering_accuracy(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering accuracy for clusters.

    :math:`\mathcal{A}, \mathcal{B}, ...` are ground truth classes and :math:`\mathcal{A}^\prime, \mathcal{B}^\prime,
    ...` are predicted classes and :math:`k` is the number of classes:

    .. math::
        \text{clustering accuracy} = 1 - \frac{1}{k} \left(\sum_{\mathcal{A}} \sum_{\mathcal{A}^\prime \neq \mathcal{B}^\prime}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{B}^\prime \cap \mathcal{A}|}{|\mathcal{A}| |\mathcal{A}|} +
         \sum_{\mathcal{A}^\prime} \sum_{\mathcal{A} \neq \mathcal{B}}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{A}^\prime \cap \mathcal{B}|}{|\mathcal{A}| |\mathcal{B}|} \right)

    This metric is proposed by `"Wang et al. Clustering-aware Multiple Graph Matching via Decayed Pairwise Matching
    Composition. AAAI 2020." <https://ojs.aaai.org/index.php/AAAI/article/view/5528/5384>`_

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering accuracy
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    batch_num = pred_clusters.shape[0]

    gt_classes_t = []

    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_acc = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        sum = 0
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                pred_i = (pred_clusters[b] == i).to(dtype=torch.float)
                gt_j = (gt_clusters[b] == j).to(dtype=torch.float)
                gt_k = (gt_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(pred_i * gt_j) * torch.sum(pred_i * gt_k)) / torch.sum(pred_i) ** 2
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                gt_i = (gt_clusters[b] == i).to(dtype=torch.float)
                pred_j = (pred_clusters[b] == j).to(dtype=torch.float)
                pred_k = (pred_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(gt_i * pred_j) * torch.sum(gt_i * pred_k)) / (torch.sum(pred_j) * torch.sum(pred_k))

        cluster_acc[b] = 1 - sum / num_clusters[b].to(dtype=torch.float)

    return cluster_acc

def clustering_purity(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering purity for clusters.

    :math:`n` is the number of instances,
    :math:`\mathcal{C}_i` represent the predicted class :math:`i` and :math:`\mathcal{C}^{gt}_j` is ground truth class :math:`j`:

    .. math::
        \text{clustering purity} = \frac{1}{n} \sum_{i=1}^{k} \max_{j\in\{1,...,k\}} |\mathcal{C}_i \cap \mathcal{C}^{gt}_{j}|

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_purity = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        for i in range(num_clusters[b]):
            max_counts = torch.max(torch.unique(gt_clusters[b][pred_clusters[b] == i], return_counts=True)[-1]).to(dtype=torch.float)
            cluster_purity[b] += max_counts / num_instances

    return cluster_purity


def rand_index(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Rand index measurement for clusters.

    Rand index is computed by the number of instances predicted in the same class with the same label :math:`n_{11}` and
    the number of instances predicted in separate classes and with different labels :math:`n_{00}`, normalized by the total
    number of instances pairs :math:`n(n-1)`:

    .. math::
        \text{rand index} = \frac{n_{11} + n_{00}}{n(n-1)}

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)
    pred_pairs = pred_clusters.unsqueeze(-1) == pred_clusters.unsqueeze(-2)
    gt_pairs = gt_clusters.unsqueeze(-1) == gt_clusters.unsqueeze(-2)
    unmatched_pairs = torch.logical_xor(pred_pairs, gt_pairs).to(dtype=torch.float)
    rand_index = 1 - torch.sum(unmatched_pairs, dim=(-1,-2)) / (num_instances * (num_instances - 1))
    return rand_index






def eval(bm, prediction, classes, verbose=False, rm_gt_cache=True):
        r"""
        Evaluate test results and compute matching accuracy and coverage.

        :param prediction: list, prediction result, like ``[{'ids': (id1, id2), 'cls': cls, 'permmat': np.array or scipy.sparse}, ...]``
        :param classes: list of evaluated classes
        :param verbose: bool, whether to print the result
        :param rm_gt_cache: bool, whether to remove ground truth cache
        :return: evaluation result in each class and their averages, including p, r, f1 and their standard deviation and coverage

        .. note::
            If there are duplicate data pair in ``prediction``, this function will only evaluate the first pair and
            expect that this pair is also the first fetched pair. Therefore, it is recommended that ``prediction`` is
            built in an ordered manner, and not shuffled.

        .. note::
            Ground truth cache is saved when data pairs are fetched, and should be removed after evaluation. Make sure
            all data pairs are evaluated at once, i.e., ``prediction`` should contain all fetched data pairs.
        """

        with open(bm.data_list_path) as f1:
            data_id = json.load(f1)

        cls_dict = dict()
        pred_cls_dict = dict()
        result = dict()
        id_cache = []
        cls_precision = dict()
        cls_recall = dict()
        cls_f1 = dict()

        for cls in classes:
            cls_dict[cls] = 0
            pred_cls_dict[cls] = 0
            result[cls] = dict()
            cls_precision[cls] = []
            cls_recall[cls] = []
            cls_f1[cls] = []

        if bm.name != 'SPair71k':
            for key, obj in bm.data_dict.items():
                if (key in data_id) and (obj['cls'] in classes):
                    cls_dict[obj['cls']] += 1
        else:
            for cls in classes:
                cls_dict[cls] = bm.compute_img_num([cls])[0]

                     
        for pair_dict in prediction:
            ids = (pair_dict['ids'][0], pair_dict['ids'][1])
            if ids not in id_cache:
                id_cache.append(ids)
                pred_cls_dict[pair_dict['cls']] += 1
                perm_mat = pair_dict['perm_mat']
                gt_path = os.path.join(bm.gt_cache_path, str(ids) + '.npy')
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_array = gt.toarray()
                assert type(perm_mat) == type(gt_array)

                assert gt_array.sum() != 0, 'ground truth permutation matrix should not be all zeros'
                
                
                
                if perm_mat.sum() == 0:
                    precision = 0
                    recall = 0
                else:
                    precision = (perm_mat * gt_array).sum() / perm_mat.sum()
                    recall = (perm_mat * gt_array).sum() / gt_array.sum()
                if precision == 0 or recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)

                cls_precision[pair_dict['cls']].append(precision)
                cls_recall[pair_dict['cls']].append(recall)
                cls_f1[pair_dict['cls']].append(f1_score)

          
        p_sum = 0
        r_sum = 0
        f1_sum = 0
        p_std_sum = 0
        r_std_sum = 0
        f1_std_sum = 0

       
        for cls in classes:
            result[cls]['precision'] = np.mean(cls_precision[cls])
            result[cls]['recall'] = np.mean(cls_recall[cls])
            result[cls]['f1'] = np.mean(cls_f1[cls])
            result[cls]['precision_std'] = np.std(cls_precision[cls])
            result[cls]['recall_std'] = np.std(cls_recall[cls])
            result[cls]['f1_std'] = np.std(cls_f1[cls])
            result[cls]['coverage'] = 2 * pred_cls_dict[cls] / (cls_dict[cls] * (cls_dict[cls] - 1))
            p_sum += result[cls]['precision']
            r_sum += result[cls]['recall']
            f1_sum += result[cls]['f1']
            p_std_sum += result[cls]['precision_std']
            r_std_sum += result[cls]['recall_std']
            f1_std_sum += result[cls]['f1_std']

        result['mean'] = dict()
        result['mean']['precision'] = p_sum / len(classes)
        result['mean']['recall'] = r_sum / len(classes)
        result['mean']['f1'] = f1_sum / len(classes)
        result['mean']['precision_std'] = p_std_sum / len(classes)
        result['mean']['recall_std'] = r_std_sum / len(classes)
        result['mean']['f1_std'] = f1_std_sum / len(classes)

        if verbose:
            print('Matching accuracy')
            for cls in classes:
                print('{}: {}'.format(cls, 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}, cvg = {:.4f}' \
                                      .format(result[cls]['precision'], result[cls]['precision_std'],
                                              result[cls]['recall'], result[cls]['recall_std'], result[cls]['f1'],
                                              result[cls]['f1_std'], result[cls]['coverage']
                                              )))
            print('average accuracy: {}'.format('p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
                                                .format(result['mean']['precision'], result['mean']['precision_std'],
                                                        result['mean']['recall'], result['mean']['recall_std'],
                                                        result['mean']['f1'], result['mean']['f1_std']
                                                        )))
        if rm_gt_cache:
            bm.rm_gt_cache(last_epoch=False)
      
        return result

def eval_cls(bm, prediction, cls, verbose=False):
        r"""
        Evaluate test results and compute matching accuracy and coverage on one specified class.

        :param prediction: list, prediction result on one class, like ``[{'ids': (id1, id2), 'cls': cls, 'permmat': np.array or scipy.sparse}, ...]``
        :param cls: str, evaluated class
        :param verbose: bool, whether to print the result
        :return: evaluation result on the specified class, including p, r, f1 and their standard deviation and coverage

        .. note::
            If there are duplicate data pair in ``prediction``, this function will only evaluate the first pair and
            expect that this pair is also the first fetched pair. Therefore, it is recommended that ``prediction`` is
            built in an ordered manner, and not shuffled. Same as the function ``eval``.

        .. note::
            This function will not automatically remove ground truth cache. However, you can still mannually call the
            class function ``rm_gt_cache`` to remove groud truth cache after evaluation.
        """

        with open(bm.data_list_path) as f1:
            data_id = json.load(f1)

        result = dict()
        id_cache = []
        cls_precision = []
        cls_recall = []
        cls_f1 = []

        cls_dict = 0
        pred_cls_dict = 0

        if bm.name != 'SPair71k':
            for key, obj in bm.data_dict.items():
                if (key in data_id) and (obj['cls'] == cls):
                    cls_dict += 1
        else:
            cls_dict = bm.compute_img_num([cls])[0]

        for pair_dict in prediction:
            ids = (pair_dict['ids'][0], pair_dict['ids'][1])
            if ids not in id_cache:
                id_cache.append(ids)
                pred_cls_dict += 1
                perm_mat = pair_dict['perm_mat']
                gt_path = os.path.join(bm.gt_cache_path, str(ids) + '.npy')
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_array = gt.toarray()
                assert type(perm_mat) == type(gt_array)

                assert gt_array.sum() != 0, 'ground truth permutation matrix should not be all zeros'
                if perm_mat.sum() == 0:
                    precision = 0
                    recall = 0
                else:
                    precision = (perm_mat * gt_array).sum() / perm_mat.sum()
                    recall = (perm_mat * gt_array).sum() / gt_array.sum()
                if precision == 0 or recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)

                cls_precision.append(precision)
                cls_recall.append(recall)
                cls_f1.append(f1_score)

        result['precision'] = np.mean(cls_precision)
        result['recall'] = np.mean(cls_recall)
        result['f1'] = np.mean(cls_f1)
        result['precision_std'] = np.std(cls_precision)
        result['recall_std'] = np.std(cls_recall)
        result['f1_std'] = np.std(cls_f1)
        result['coverage'] = 2 * pred_cls_dict / (cls_dict * (cls_dict - 1))

        if verbose:
            print('Class {}: {}'.format(cls, 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}, cvg = {:.4f}' \
                                        .format(result['precision'], result['precision_std'], result['recall'],
                                                result['recall_std'], result['f1'], result['f1_std'], result['coverage']
                                                )))
            
      
        return result