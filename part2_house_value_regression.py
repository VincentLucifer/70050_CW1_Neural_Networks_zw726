import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Regressor:

    def __init__(self, x, nb_epoch = 1000, learning_rate = 1e-3,
                 batch_size = 128, hidden_sizes = (128, 64, 32),
                 weight_decay = 1e-6, seed = 0):
        """ 
        Initialise the regression model.

	You can add any input parameters you need.
        Just remember to set them with default values, so the LabTS tests still work.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.weight_decay = weight_decay
        self.seed = seed
        self.output_size = 1

        X, _ = self._preprocessor(self._augment(x), training = True)
        self.input_size = X.shape[1]
        self._build_model()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _augment(self, x):
        """
        Add ratio features when the housing columns are present.
        Copies the frame so the caller's data is left unchanged.
        """
        X = x.copy()
        if {"total_rooms", "households"}.issubset(X.columns):
            households = X["households"].replace(0, np.nan)
            X["rooms_per_household"] = X["total_rooms"] / households
            if "population" in X.columns:
                X["population_per_household"] = X["population"] / households
        if {"total_bedrooms", "total_rooms"}.issubset(X.columns):
            rooms = X["total_rooms"].replace(0, np.nan)
            X["bedrooms_per_room"] = X["total_bedrooms"] / rooms

        # Counts are heavily skewed. log1p gives the network a second, better scaled view.
        for col in (
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ):
            if col in X.columns:
                X[col + "_log"] = np.log1p(X[col].clip(lower=0))

        # Distance to major California cities. Location is not linear in lat/long alone.
        if {"latitude", "longitude"}.issubset(X.columns):
            cities = {
                "los_angeles": (34.05, -118.24),
                "san_francisco": (37.77, -122.42),
                "san_diego": (32.72, -117.16),
                "san_jose": (37.33, -121.89),
                "sacramento": (38.58, -121.49),
                "fresno": (36.74, -119.79),
            }
            lat = X["latitude"].to_numpy(dtype=float)
            lon = X["longitude"].to_numpy(dtype=float)
            for name, (city_lat, city_lon) in cities.items():
                X["dist_" + name] = np.sqrt((lat - city_lat) ** 2 + (lon - city_lon) ** 2)
        return X

    def _build_model(self):
        """Create the MLP used for regression. Linear output, ReLU hidden layers."""
        layers = []
        in_dim = self.input_size
        for hidden in self.hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(torch.nn.ReLU())
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, self.output_size))
        self.model = torch.nn.Sequential(*layers)
        self.model.eval()

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        # Build a copy of the dataframes
        given_y = isinstance(y, pd.DataFrame)
        X = x.copy()
        if given_y:
            Y = y.copy()

        # Set parameters for preprocessor
        if training :
            self.preproc_text = {}
            pre_X = X.select_dtypes(include=['float'])
            self.preproc_fill = pd.DataFrame( [pre_X.median().to_list(), \
                                       pre_X.min().to_list(), \
                                       pre_X.max().to_list()],\
                                      columns = pre_X.columns.to_list())
            if given_y:
                self.preproc_fill_out = [Y.iloc[:, 0].median(), \
                                 Y.iloc[:, 0].min(), \
                                 Y.iloc[:, 0].max()]

        # Replace missing values in X
        X.fillna(self.preproc_fill.iloc[0], inplace=True) 

        # Replace textual values in X
        for column in X.select_dtypes(include=['object']).columns:
            lb = preprocessing.LabelBinarizer()
            if training :
                lb.fit(X[column])
                self.preproc_text[column] = lb.classes_
            else:
                lb.classes_ = self.preproc_text[column]
            values = lb.transform(X[column])
            X = X.loc[:, X.columns != column] # Remove current column
            for new_col in range (len(lb.classes_)): # Add new ones
                X.insert(len(X.columns), column + "_" + str(new_col), values[:, new_col])

        # Normalise
        for column in X.select_dtypes(include=['float']).columns:
            X[column] = X[column].subtract(self.preproc_fill[column].iloc[1]) \
                .div(self.preproc_fill[column].iloc[2] - self.preproc_fill[column].iloc[1])
        if given_y:
            if (Y.iloc[:, 0] > 1).any():
                Y.iloc[:,0] = Y.iloc[:,0].subtract(self.preproc_fill_out[1]) \
                    .div(self.preproc_fill_out[2] - self.preproc_fill_out[1])

        return torch.tensor(X.values.astype('float'), dtype = torch.float), \
               torch.tensor(Y.values.astype('float'), dtype = torch.float) if given_y \
                                                                           else None


    def _revert_preprocessor(self, y):
        """ 
        Reverse the normalisation of the targets.
          
        Arguments:
            y {torch.tensor} -- Raw target array of shape (batch_size, 1).

        Returns:
            {np.ndarray} -- Unnormalised target values (batch_size, 1).
        """
        y = y * (self.preproc_fill_out[2] - self.preproc_fill_out[1]) + self.preproc_fill_out[1]
        return y.detach().numpy()


    def fit(self, x, y):
        """
        Regressor training function

        Performs the following steps number-of-epochs times:
            - Performs forward pass through the model given the input.
            - Computes the loss based on this forward pass.
            - Performs backwards pass to compute gradients of loss with 
              respect to parameters of the model.
            - Performs one step of gradient descent on the model parameters.
        You are free to implement any additional steps to improve 
              learning (batch-learning, shuffling,...).

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X, Y = self._preprocessor(self._augment(x), y = y, training = True)
        self.input_size = X.shape[1]
        if self.model[0].in_features != self.input_size:
            self._build_model()

        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = max(self.nb_epoch, 1)
        )
        loss_fn = torch.nn.MSELoss()

        n = X.shape[0]
        batch_size = max(int(self.batch_size), 1)
        for epoch in range(self.nb_epoch):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = X[idx]
                yb = Y[idx]
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            scheduler.step()
            if epoch % 50 == 0 or epoch == self.nb_epoch - 1:
                print(
                    "Epoch {}: training loss = {:.6f}".format(
                        epoch, total_loss / max(n_batches, 1)
                    )
                )

        self.model.eval()
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x using the regressor model.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).
        
         Hint:
            You can use the provided _preprocessor and _revert_preprocessor 
            functions.
        
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(self._augment(x), training = False)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
        return self._revert_preprocessor(preds)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def evaluate(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.
        This should call predict(), which itselt should call the preprocessor.
        Returns RMSE for the regression model on the given dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- True target values. Array of shape (batch_size, 1).

        Returns:
            {float} -- Returns RMSE on this dataset. RMSE(f(x), y)

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        preds = np.asarray(self.predict(x), dtype = float).reshape(-1)
        targets = np.asarray(y, dtype = float).reshape(-1)
        return float(np.sqrt(np.mean((preds - targets) ** 2)))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor.
    # Running this file as a script would otherwise pickle the class as __main__.Regressor,
    # which LabTS cannot import. Point the class at this module before saving.
    trained_model.__class__.__module__ = "part2_house_value_regression"
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def main():
    """
    The main function to train, save and evaluate the regression model.
    This function is not called by LabTS, so feel free to change it.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Hold out a validation split so the printed error is not the training error.
    # The model saved for submission is trained on the full dataset afterwards.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_train))
    split = int(0.8 * len(perm))
    train_idx = perm[:split]
    val_idx = perm[split:]

    x_tr = x_train.iloc[train_idx].reset_index(drop = True)
    y_tr = y_train.iloc[train_idx].reset_index(drop = True)
    x_val = x_train.iloc[val_idx].reset_index(drop = True)
    y_val = y_train.iloc[val_idx].reset_index(drop = True)

    probe = Regressor(x_tr, nb_epoch = 200)
    probe.fit(x_tr, y_tr)
    val_error = probe.evaluate(x_val, y_val)
    print(f"\nValidation RMSE (probe): {val_error}\n")

    regressor = Regressor(x_train, nb_epoch = 200)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    train_error = regressor.evaluate(x_train, y_train)
    print(f"\nSaved model training RMSE: {train_error}\n")


if __name__ == "__main__":
    main()

