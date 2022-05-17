general approach to forecasting:

let xval be the dataset we wish to forecast over
let n be the number of rows in xval
let the lag parameter be hardcoded to be 10
let c be the number of channels in xval

create a dataframe containing (n + 10) rows (the last 10 rows of the training dataset prepended to xval)
for one iteration do the following
(1) from trace.posterior, sample 1 value at random for each data variable
(2) for each channel, calculate the orders contribution: coef_parameter_value * sat(car(column_raw_spends, car_parameter_value), sat_parameter_value)
    this returns c arrays of (n+10) values
(3) sum up each contribution to get a single array of (n+10) values

repeat above steps (1-3) for as many iterations as you'd like (~1000) and average

