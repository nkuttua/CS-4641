import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        return np.sqrt(np.square(pred - label).mean())

    def construct_polynomial_feats(self, x: np.ndarray, degree: int) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        N, D = x.shape
        if D == 1:
            feat = np.zeros((N, degree + 1))
        else:
            feat = np.zeros((N, degree + 1, D))
        for n in range(N):
            for d in range(degree + 1):
                if D == 1:
                    feat[n, d] = np.power(x[n], d)
                else:
                    feat[n, d, :] = np.power(x[n, :], d)
        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.dot(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        a = np.linalg.pinv(np.dot(xtrain.T, xtrain))
        b = np.dot(a, xtrain.T)
        weight = np.dot(b, ytrain)
        return weight

    def linear_fit_GD(self,xtrain: np.ndarray,ytrain: np.ndarray,epochs: int = 5,learning_rate: float = 0.001,) -> Tuple[np.ndarray, List[float]]:  # [5pts] BONUS
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weights = np.zeros((xtrain.shape[1], 1))
        loss_per_epoch = []

        for i in range(epochs):
            y_pred = xtrain.dot(weights)
            error = y_pred - ytrain
            gradient = xtrain.T.dot(error) / xtrain.shape[0]
            weights = weights - learning_rate * gradient
            rmse_loss = self.rmse(ytrain, y_pred)
            loss_per_epoch.append(rmse_loss)
        return weights, loss_per_epoch

    def linear_fit_SGD(self,xtrain: np.ndarray,ytrain: np.ndarray,epochs: int = 100,learning_rate: float = 0.001,) -> Tuple[np.ndarray, List[float]]:  # [5pts] BONUS
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        raise NotImplementedError

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        N, D = xtrain.shape
        I = np.eye(D)
        I[0][0] = 0
        return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(xtrain), xtrain) + c_lambda * I), np.transpose(xtrain)), ytrain)

    def ridge_fit_GD(self,xtrain: np.ndarray,ytrain: np.ndarray,c_lambda: float,epochs: int = 500,learning_rate: float = 1e-7,) -> Tuple[np.ndarray, List[float]]:  # [5pts] BONUS
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        raise NotImplementedError

    def ridge_fit_SGD(self,xtrain: np.ndarray,ytrain: np.ndarray,c_lambda: float,epochs: int = 100,learning_rate: float = 0.001,) -> Tuple[np.ndarray, List[float]]:  # [5pts] BONUS
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        raise NotImplementedError

    def ridge_cross_validation(self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        fold_size = int(X.shape[0] / kfold)
        loss_per_fold = []
        for i in range(kfold):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            x_test_fold = X[test_start:test_end]
            y_test_fold = y[test_start:test_end]
            x_train_fold = np.concatenate((X[:test_start], X[test_end:]))
            y_train_fold = np.concatenate((y[:test_start], y[test_end:]))
            weights = self.ridge_fit_closed(x_train_fold, y_train_fold, c_lambda)
            y_pre_fold = x_test_fold @ weights
            loss = self.rmse(y_test_fold, y_pre_fold)
            loss_per_fold.append(loss)
        return loss_per_fold


    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS
        
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            error_list.append(err)
            if best_error is None or err < best_error:
                best_error = err
                best_lambda = lm

        return best_lambda, best_error, error_list
