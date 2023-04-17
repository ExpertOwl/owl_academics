# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:38:50 2020

@author: Zanon
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:18:17 2020

@author: Zanon
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:31:50 2020

@author: Zanon
"""
# import numpy as np
# from os import listdir, path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest, tee, count
from  os import listdir
from collections import defaultdict
from scipy.optimize import curve_fit
# =============================================================================
# These dicitonaries contain regex keys that when assembled, can parse
# files text files for data. (sorry! I don't know how else to do it!)
# There is one for the weights table and one for the energy table.
# Complicated, but e.g:...
#     k is one or two digits -> \d{1,2}
#     n_v and n_o are one or more digits -> \d+
#     weight(o)  = x.xxxxxx -> digit period 6*digit -> \d.\d{6}
#     disc-weight(o) = x.xxxxxxE±xx
#                       -> [6xdigit]E±[2x digits] -> \d\.\d{6}E[-,\+]\d{2}.
#
#     The final regex is bult as
#          f'{regs["tab"]}'.join([reg for reg in regs.values()])
#     i.e. tab k tab eta_k tab tau_k tab n_o tab...
#
# f strings and r strings made this implementaion very practical and 'easy' to 
# update if the tables change.
# The table headers are the dicitonary keys which retain order after ~3.6
# =============================================================================
#
#
# The long and short are these are crazy and unreadable but they work, if 
# you need to change the table or have other issues you can give me a shout and
# I can help you work through it
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


out0list = [file for file in listdir() if file.endswith('.out0')]
# out0list = ['Benzene.out0']

def sliding_window(iterable, n=2):
    iterables = tee(iterable, n)
    for iterable, num_skipped in zip(iterables, count()):
        for _ in range(num_skipped):
            next(iterable, None)
    return zip(*iterables)


def parse_files(out0list):
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
    # returns sum of p*x**i,
    # coeffs=[a,b,c] -> ax^0 + bx^1 + cx^2
    return(sum([p*(x**i) for i, p in enumerate(coeffs)]))

def linear(x, m, b):
    return(m*x+b)

def ln_gamma_fn(x, b, gamma):
    return(b*np.log((1+gamma*x)))


data = parse_files(out0list)
errors_dict = defaultdict(list)
mp_last_errors = defaultdict(list)
# start, stop = (10, 15)
start = 0
stop = -4
skipped = 0
results = list()
for mol_name, mol_data in data.groupby('mol'):
    print(mol_name)
    mol_data = mol_data.drop_duplicates()
    energies = mol_data[['MP', 'CC']]
    xvals = mol_data['disc-weight(v)']
    xfit = xvals.iloc[start:stop]
    yfit = energies['CC'][start:stop]

    try:
        b_init = yfit.iloc[-1]
        m_init = (yfit.iloc[-1] - yfit.iloc[0])/(xfit.iloc[-1] - xfit.iloc[0])
        answers = energies.iloc[-1]
        (m,b), cov = curve_fit(linear, xfit, yfit,
                                  p0 = [m_init, b_init])
    except Exception as e:
        skipped+=1
        print(e)
        continue
    energy_prediction = b
    pred_error = b - answers['CC']
    results.append([mol_name, answers['CC'], energy_prediction,pred_error,mol_data['percent_virt'][xfit.keys()].iloc[-1]])
      #######PLOTTING######## 
    fig,ax = plt.subplots()
    ax.plot(xvals,energies['CC'],label='data', marker='x', linestyle=':')
    ax.plot(xfit,yfit, label='fitting points', marker='x', linestyle=' ')
    ax.plot(np.linspace(xvals.iloc[0],0),
             linear(np.linspace(xvals.iloc[0],0),m,b),
             label='fit(linear)')
    ax.plot([], [], '-r', label = r'% Vrt', linestyle=' ',
             marker='s', color='grey')
    ax.set_ylabel(r'$E_{Corr}$')
    ax.set_xlabel(r'Weight of Discarded Virtual Orbitals')
    plt.legend(loc='center left')
    ax2=ax.twinx()
    ax2.plot(xfit, mol_data['percent_virt'][xfit.keys()],label = r'% Vrt', linestyle=' ',
             marker='s', color='grey')
    ax2.set_ylabel(r'Virtual Orbitals Used (%)')
    plt.title(f'{mol_name}, Extrapolation Over Discarded Virtual Orbitals \n Error = {float(pred_error):.2E} Hartree')
    
    # for label in energies:
    #     (b, gamma), cov = curve_fit(ln_gamma_fn, xfit,
    #                                 errors[label].iloc[start:stop],
    #                                 maxfev=5000)
    #     new_x = ln_gamma_fn(xvals, b, gamma)
    #     new_xfit = ln_gamma_fn(xfit, b, gamma)
    #     (e_0, e_1), energy_cov = curve_fit(arbitrary_poly, new_xfit,
    #                                        energies[label].iloc[start:stop],
    #                                        p0=[1,1])
    #     energy_predictions = arbitrary_poly(new_x, e_0, e_1)
    #     pred_error = energy_predictions.iloc[-1] - answers[label]
    #     errors_dict[mol_name].append(pred_error)

# errors_dict = pd.DataFrame.from_dict(errors_dict)
# mp_last_errors = pd.DataFrame.from_dict(mp_last_errors)
# mp_last_errors.index = ['MP + CC']
# errors_dict.index = [label + ' extrap' for label in energies.keys()]
# errors_dict = pd.concat([errors_dict,mp_last_errors])


results=pd.DataFrame(results)
results.index=results[0]
results=results.drop(columns=0)
results.columns=('Actual', 'Predicted', 'Error', '%used')
print(results)

results = results.T
results[0] = [0,0,0,0]
results= results.T
print(abs(results).describe())
print(f'skipped {skipped} files')
with open('energies.csv','w') as file:
    results.to_csv(file.name)