# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:12:47 2021

@author: Zanon
"""
import pandas as pd
from collections import defaultdict

# with open('ln.csv','r') as file:
with open('energies.csv','r') as file:
    calc_energies = pd.read_csv(file, index_col=0)
    calc_energies=calc_energies.T
with open('reactions.csv','r') as file:
    reactions = pd.read_csv(file, index_col = 0,dtype=int)    


results = list()
# reactions 
for i in range(len(reactions)):
    rxn = reactions.iloc[i]   
    mols = rxn.iloc[0:4]
    stoich = rxn.iloc[4:]
    stoich.index = mols
    try:
        extrap_nrg = calc_energies[mols.values].T['Predicted_Energy']
        actual_nrg = calc_energies[mols.values].T['Actual_Energy']
    except Exception as e:
        print(e)
        continue
    extrap_nrg = sum(extrap_nrg * stoich)
    actual_nrg = sum(actual_nrg * stoich)
    error = extrap_nrg - actual_nrg 
    
    results.append([i,actual_nrg,extrap_nrg, error])
    
results=pd.DataFrame(results)
results.index=results[0]
results=results.drop(columns=0)
results.columns=('Actual', 'Predicted', 'Error')
    
print(results)
print(abs(results).describe())

    