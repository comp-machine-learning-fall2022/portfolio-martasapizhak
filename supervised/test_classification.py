import numpy as np
import funcs_classification

hw_data = np.genfromtxt("data_classification.csv", delimiter=',')


def test_mean_centered_shape():
	out = funcs_classification.mean_centered(hw_data)
	assert out.shape == hw_data.shape


def test_mean_centered_col3_unique():
	out = funcs_classification.mean_centered(hw_data)
	unique_labs = np.unique(out[:,2])
	assert len(unique_labs) == 2


def test_mean_centered_col3_vals():
	out = funcs_classification.mean_centered(hw_data)
	unique_labs = np.unique(out[:,2])
	assert (0,1) == (min(unique_labs), max(unique_labs))


def test_mean_centered_mean():
	out = funcs_classification.mean_centered(hw_data)
	assert np.isclose(np.sum(np.mean(out[:,:2], axis=0)),0)


def test_tenfold_type():
	CV = funcs_classification.ten_fold(hw_data, 'rbf', 0)
	assert isinstance(CV, float) 


# Parts of test file updated based on questions and 
#               corrections by M. Litz (Smith 2020)
# Additional edits by K. Hablutzel (Smith 2023 J)
