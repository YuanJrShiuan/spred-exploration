import time
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Lasso, LassoLars

from models import SparedLinearRegression
from utils import eval_over_linear_regression_datasets


def run_lasso(alpha, x, y, method='default', **kwargs):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    if method == 'LARS':
        lasso_regressor = LassoLars(alpha=alpha, normalize=False, max_iter=100000)
    else:
        lasso_regressor = Lasso(alpha=alpha, max_iter=100000)

    t = time.time()
    lasso_regressor.fit(x, y)
    t = time.time() - t

    _coef = lasso_regressor.coef_.reshape(respond_dim, predictor_dim)
    weights = _coef.T

    return {'time': t,
            'weights': weights}


def run_rs_regression(alpha, x, y,
                      net=None,
                      optname='SGD',
                      epochs=200,
                      batch_size=512,
                      lr=1e-4,
                      loss_func='ce',
                      device='cuda:0',
                      eval_every_epoch=100, **kwargs):

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    if loss_func == 'ce':
        y_tensor = torch.tensor(y, dtype=torch.int64, device=device)
    elif loss_func == 'mse':
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    if net is None:
        _, predictor_dim = x.shape
        _, respond_dim = y.shape
        model = SparedLinearRegression(input_dim=predictor_dim,
                                       output_dim=respond_dim)
    else:
        model = net

    # using weight decay for L2 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr, weight_decay=alpha)

    metric_list = []
    t = time.time()
    with trange(epochs) as titer:
        for e in titer:
            metric = {}
            total_loss = 0
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                weight_dict = model.get_weights()
                l1_reg = 0
                loss = 0
                for k, w in weight_dict.items():
                    l1_reg += torch.norm(w, p=1)
                if loss_func == 'mse':
                    _func = torch.nn.MSELoss()
                    loss += _func(y_pred, y_batch)
                elif loss_func == 'ce':
                    _func = torch.nn.CrossEntropyLoss()
                    loss += _func(y_pred, y_batch)
                assert not torch.isnan(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() + alpha * l1_reg.item()

            epoch_loss = total_loss / len(dataloader)
            metric['epoch_loss'] = epoch_loss
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t


            if (e + 1) % eval_every_epoch == 0:
                m = eval_over_linear_regression_datasets(x, y, model, alpha)
                metric.update(m)
                metric_list.append(metric)

            titer.set_postfix(metric)

    t = time.time() - t

    weights = model.get_weights()['weight'].reshape([predictor_dim, respond_dim])

    return {'time': t,
            'weights': weights,
            'metric_list': metric_list}


def run_gradient_descent_lasso(alpha, x, y, max_iter=10000, lr=1e-3, tol=1e-6):
    """
    Solve the Lasso regression problem using gradient descent.
    :param alpha: L1 regularization coefficient
    :param x: Feature matrix
    :param y: Target values
    :param max_iter: Maximum number of iterations
    :param lr: Learning rate
    :param tol: Convergence tolerance
    :return: Trained weights, training time, number of iterations, and other information
    """
    import numpy as np
    import time

    n_samples, n_features = x.shape
    # Initialize weights
    weights = np.zeros(n_features)
    # Convert data types
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64).flatten()  # Ensure y is a 1D array
    # Record training time
    start_time = time.time()
    # Normalize features
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std
    y_mean = np.mean(y)
    y = y - y_mean
    # Gradient descent iteration
    for iter in range(max_iter):
        # Compute predictions
        y_pred = np.dot(x, weights)
        # Compute error
        error = y_pred - y
        # Compute gradient
        grad = (np.dot(x.T, error) / n_samples) + alpha * np.sign(weights)
        # Update weights
        weights -= lr * grad
        # Check convergence condition
        if np.linalg.norm(grad) < tol:
            break
    # Calculate training time
    train_time = time.time() - start_time
    # Return results
    return {
        'weights': weights,
        'time': train_time,
        'iterations': iter + 1
    }
