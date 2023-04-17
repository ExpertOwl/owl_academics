# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:23:54 2021

@author: Zanon
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:38:50 2020

@author: Zanon
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest, tee, count
from  os import listdir
from collections import defaultdict
from scipy.optimize import curve_fit

#I use regex to parse the files and pull tables:  
# =============================================================================
# These dicitonaries contain regex keys that when assembled, can parse
# There is one for the weights table and one for the energy tables.
# f strings strings make this implementaion very practical and easy to 
# update if the tables change.
# The table headers are the dicitonary keys. Need python >= 3.6 so they retain
# order
# =============================================================================
tab = r"\s+"
weight_table_regs = {'k': r'\d{1,2}',
                     'eta_k': r'\d{1,2}\.\d{3}',
                     'tau_k': r'\d\.\d{4}E[-,\+]\d{2}',
                     'n_o': r'\d+',
                     'weight(o)': r'\d\.\d{6}',
                     'disc-weight(o)': r'\d\.\d{6}E[-,\+]\d{2}',
                     'n_v': r'\d+',
                     'weight(v)': r'\d\.\d{6}',
                     'disc-weight(v)': r'\d\.\d{6}E[-,\+]\d{2}',
                     'disc-weight(t)': r'\d\.\d{6}E[-,\+]\d{2}'}

energy_table_regs = {'k': r'\d{1,2}',
                     'eta_k': r'\d{1,2}\.\d{3}',
                     'tau_k': r'\d\.\d{6}E[-,\+]\d{2}',
                     'E_corr(T1)': r'\-*\d.\d{10}',
                     'E_corr(T2)': r'\-*\d.\d{10}',
                     'E_corr': r'\-*\d.\d{10}',
                     }

weight_table_regex = re.compile(tab.join(
    [reg for reg in weight_table_regs.values()]))
energy_table_regex = re.compile(tab.join(
    [reg for reg in energy_table_regs.values()]))


def raw_text_to_table(text):
    # converts a table of text to a list of list of floats
    # "a1 a2 ... a_n
    # b1 b2 ... b_n" -> [[a1,a2,...],[b1,b2,...]]
    acc = []
    for line in text:
        line = line.strip().split()
        line = [float(num) for num in line]
        acc.append(line)
    return(acc)

def sliding_window(iterable, n=2):
    #Generates a sliding window of length n over an iterable eg:
    # sliding_window([1,2,3,4], n=2) -> [1,2], [2,3], [3,4] 
    iterables = tee(iterable, n)
    for iterable, num_skipped in zip(iterables, count()):
        for _ in range(num_skipped):
            next(iterable, None)
    return zip(*iterables)


def parse_files(out0list):
    #This function pulls tables from the file using regex and then 
    #puts it into a pandas dataframe
    list_of_frames = []
    for file in out0list:
        print(f'file = {file}')
        with open(file, 'r') as mol_file:
            mol_name = file.split('.')[0]
            text = mol_file.read()
            weight_text = re.findall(weight_table_regex, text)
            energy_text = re.findall(energy_table_regex, text)
        weights = raw_text_to_table(weight_text)
        data = pd.DataFrame(weights, columns=weight_table_regs)
        data = data.drop_duplicates()
        data = data.drop(labels=['k', 'eta_k', 'tau_k'], axis=1)
        energies = raw_text_to_table(energy_text)
        # build an iterator of energies because its simple to divide up
        itergies = zip_longest(*[iter(energies)]*len(data))
        itergies = [pd.DataFrame(table, columns=energy_table_regs)['E_corr']
                    for table in itergies]
        if len(itergies) > 1:
            CC_MP, MP, CC = itergies
            data['CC'] = CC
            data['MP'] = MP
            data['CC_MP'] = CC_MP
        else:
            data['CC_MP'] = itergies[0]
        data['percent_virt'] = data['n_v']/data['n_v'].iloc[-1]
        data['mol'] = mol_name
        data = data.drop(0)
        list_of_frames.append(data)
    formatted_df = pd.concat(list_of_frames)
    return(formatted_df)


def arbitrary_poly(x, *coeffs):
    #Arbitrary polynomial, order = length of coeffs
    # returns sum of p*x**i,
    # coeffs=[a,b,c] -> ax^0 + bx^1 + cx^2
    return(sum([p*(x**i) for i, p in enumerate(coeffs)]))

def linear(x, m, b):
    #y=mx+b used when I was trying machine learning for scikit learn pipeline
    return(m*x+b)

def ln_gamma_fn(x, b, gamma,c):
    return(c+b*(np.log((1+gamma*x))))



#Generate list of files. Script will grab any files with an out0 extension
out0list = [file for file in listdir() if file.endswith('.out0')]

data = parse_files(out0list)

results = list()
#empty list to hold skipped files (Files that throw an error)
skipped=[]
#Weather to show plots
show_fig = True

scheme={'data_name':'CC', #data to extrapolate on, can be CC MP or CC_MP
        'start':4, #Start index
        'stop':8, #Stop index
        'func':'ln'
        }


for mol_name, mol_data in data.groupby('mol'):
    start = scheme['start']
    stop = scheme['stop']
    data_name = scheme['data_name']
    print(mol_name)
    #Initial values for fit

    mol_data = mol_data.drop_duplicates()
    energies = mol_data[['MP', 'CC']]
    xvals = mol_data['disc-weight(v)']
    xfit = xvals[start:stop]
    yfit = energies[data_name][start:stop]
    answers = energies.iloc[-1]
    if scheme['func'] == 'ln':
    #Function used is c+b*ln(gamma*x+1)
        func = ln_gamma_fn
        gamma_init=-0.01
        b_init=-1
        c_init = yfit.iloc[-1]
        init=[b_init, gamma_init,c_init]
    elif scheme['func'] == 'poly':
    #function used is a polynomial with degree determined by 
    #length of init - 1: [a,b,c] -> 2nd degree with ax^0 + bx^1 + cx^2...
        func = arbitrary_poly
        a_init = yfit.iloc[-1]
        b_init = (yfit.iloc[-1]-yfit.iloc[-2])/(xfit.iloc[-1]-xfit.iloc[-2])
        init = [a_init, b_init]
    #Try fit, skip file and append it to skipped if an error is thrown
    try:
        (b,gamma,c), cov = curve_fit(func, xfit, yfit,
                                     p0 = [init],maxfev=3000)
    except Exception as e:
        print(e)
        skipped.append(mol_name)
        continue
    #If the fit worked, get error and caluate rsquared
    energy_prediction = c
    pred_error = energy_prediction - answers['CC']

    rsquared = sum((yfit - ln_gamma_fn(xfit,b,gamma,c))**2)
    #save results to list. This list is used to generate a summary CSV 
    results.append([mol_name, gamma, b, c, pred_error])
      #######PLOTTING######## 
    if show_fig:
        fig,ax = plt.subplots()
        #Plot data
        ax.plot(xvals,energies['CC'],label='data', marker='x', linestyle=':')
        #Plot points used in fitting
        ax.plot(xfit,yfit, label='fitting points', marker='x', linestyle=' ')
        #plot predicted points based on fit
        ax.plot(np.linspace(xvals.iloc[0],0),
                 ln_gamma_fn(np.linspace(xvals.iloc[0],0),b,gamma,c),
                 label=r'$b ln(\gamma x+1)+c$')

        plt.xlabel('Weight of Discarded Virtual Orbitals')
        plt.ylabel(r'$E_{corr}$')
        #Generate second axis to plot %vrt used
        ax2=ax.twinx()
        #Create empty line object so that the %vrt legend entry 
        #is in the right spot
        ax.plot([], [], '-r', label = r'%vrt', linestyle=' ',
                 marker='s', color='grey')
        #Create legend
        plt.legend(loc='center right') 
        #plot %virtuals for points used
        ax2.plot(xfit, mol_data['percent_virt'][xfit.keys()], label = r'%vrt', linestyle=' ',
                 marker='s', color='grey')
        ax2.set_ylabel(r'Virtual orbitals used (%)')
        plt.title(fr'{mol_name}, b={b:.2E}, gamma={gamma:.2E}')
       





results=pd.DataFrame(results)
results.index=results[0]
results=results.drop(columns=0)
results.columns=('gamma', 'b','c','error')
results=results.T
# results[0] = [0,0,0,0]
results=results.T
print(results)
print(abs(results).describe())
print(f'skipped {len(skipped)} molecules: \n {skipped}')
with open('ln.csv','w') as file:
    results.to_csv(file.name)