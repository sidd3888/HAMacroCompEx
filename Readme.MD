# Monetary and Fiscal Policy Regime Changes

The code for the project is available in `ge_hank.m` under the `Code` folder. The instructions to run the code and produce the figures in the text are as follows.

The code, as uploaded, is set to solve for a positive TFP shock with an unexpected regime change. Here are the variables to modify to produce the various plots in the PDF:

- `param.RegimeExpectations`: Change between `'known'` and `'unknown'` to solve for either the perfect foresight or imperfect foresight equilibrium
- `param.RegimeChangeOn`: Set to 0 to solve for the transition paths without a regime change. Set to 1 for a regime change. Setting to 1 and solving for the transition path with the same parameters before and after $T_0$ is NOT RECOMMENDED
- `param.tfpshock_size` or `param.discountshock_size`: To instantiate a TFP or discount rate shock. I use 0.05 for TFP and 0.001 for the discount rate shocks, respectively
- `param.fiscalrule_phi` and `param.taylor_phi`: Modify to desired values to produce the plots without a regime change

Upon setting the desired parameter values, run the code in MATLAB to produce the plots. Multiple such plots (some of which are used in the text) are already available in the `Figures` folder. An example of a regime change from regime 2 to 1 is also made available for the case of a TFP shock.
