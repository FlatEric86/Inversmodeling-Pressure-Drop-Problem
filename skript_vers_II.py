import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime

# Hallo Alex,

# Sorry für die Verspätung. Ich bin wirklich halalala gerade.
# Also, was ich gerne als Output hätte, ist die Koeffizienten a und b (so genau wie möglich) aus der Gleichung: dp =a * u + b * u^2
# Die Input sind dann dp und x (in der File im Anhang).
# Wäre es möglich das, das Skript ein Output-File erzeugt? Und drin soll „a=“ und „b=“ sein. Der Name der Output-File soll sich (input-File name)_output nennen.
# Ich hoffe ich bin klar genug mit der Erklärung. Wenn nicht freue ich mich auf Fragen.

# LG,

# Mousti



### lade Trainingsdaten

model_data_path = './UInlet_DP.csv'

dP = []
U  = []
with open(model_data_path, 'r') as fin:
    line_counter = 0
    for line in fin.readlines():
        if line_counter == 0:
            line_counter += 1
            continue
        else:
            line = line.strip('\n').split('\t')
            if len(line) > 0:
                U.append(float(line[0].replace(',','.')))
                dP.append(float(line[1].replace(',','.')))

        line_counter += 1

#Konvertiere beide Listen in NumPy-Arrays
U  = np.asarray(U)
dP = np.asarray(dP)


### definiere Vorwärtsmodell Funktion: Geschwindigkeit [u] bildet ab auf  Druckverlust [dp]
def press_diff(u, coef_vec):
    return coef_vec[0]*u**2 + coef_vec[1]*u**2


### bestimme den Koeffizientenvektor durch Gradientenabstiegsmethode
## einfach optimiert

# beste manuell erzeugte Parametrisierung [l_rate = l_rate = 0.000001, eps    = 0.000005]

# learning_rate
# l_rate = 0.0000001
l_rate = 0.000001

# Konvergenzzahl (quadratischer Modellfehler als Metrik)
eps    = 0.000005

# maximale Anzahl an Iterationen
max_iter = 100000

# definiere Koeeffizentenvektor mit initialen Werten (1,1)
coef_vec = np.ones(2)


# Quadrierte Modellfehler über alle Iterationen
ERR = []

# Gradient der Zielfunktion
grad_E  = np.array([0, 0])

N_iter = 0
while True:
    if N_iter == max_iter:
        print('Modell konvergiert nicht !!!')
        break

    ### iteriere über alle Trainingsdaten
    # temporärer Modellwertvektor
    predicted_dP = np.array([])

    # initialisiere Gradientenvektor mit den Werten 1
    

    # initiale kummulierenden Modellfehler (Summe der Fehlerquadrate über alle Trainingsdaten pro Iteration)
    err = 0

    # initialisiere absoluten mittleren Modellfehler
    abs_mean_err = 0
    ### Epoche(N_iter)
    # itteriere über alle Trainingsdaten (als beschreibender Index für die Subiteration soll <i> verwendet werden)
    for u_i, dp_i in zip(U, dP):
        # berechne die i'ten Druckdifferenzwert bzgl. der Geschwindigkeit aus den Trainingsdaten
        model_dp_i = press_diff(u_i, coef_vec)

        # füge i'ten Modellwert dem Modellvektor zu 
        predicted_dP = np.append(predicted_dP, model_dp_i)
        
        # summiere dem Gradientenvektor den i'ten Term auf das korrespondierende Element auf
        grad_E[0] += (dp_i - model_dp_i)*(-1*u_i**2)
        grad_E[1] += (dp_i - model_dp_i)*(-1*u_i)

        # kummuliere quadrierten Modellfehler
        err          += (dp_i - model_dp_i)**2
        abs_mean_err += abs(dp_i - model_dp_i)

    # passe die Koeffizienten über Gradientenabstig an    
    coef_vec = coef_vec - l_rate*grad_E


    # füge der Modellfehler-Liste den aktuellen MOdellfehler zu
    ERR.append(err)


    # prüfe auf Abbruchkriterium
    if N_iter > 0 and (err < eps or N_iter == max_iter):   

        if N_iter < max_iter:
            print('\n')
            print(80*'~')
            print('Optimierung konvergierte nach    : ', str(N_iter), ' Iterationen')
            print('Mean Squared Error               : ', str(ERR[-1]))
            print('absoluter mittlerer Modellfehler : ', (abs_mean_err/len(U)))
            break
        else:
            print('Keine Konvergenz')
            sys.exit()

    N_iter += 1

Y = []
for u in U:
    Y.append(press_diff(u, coef_vec))

r2 = r2_score(np.asarray(dP), np.asarray(Y))


print(80*'~')
print('Funktionenbeschreibung           : dp := au² + bu')
print('a                                =', coef_vec[0])
print('b                                =', coef_vec[1])
print('R²                               =', r2)
print(80*'~')



### FILE OUPUT SECTION



model_data_path_out = model_data_path[:-4] + '_result.txt'

with open(model_data_path_out, 'w') as fout:
    fout.write('\n\n')
    fout.write('Funktionenbeschreibung : dp := au² + bu')
    fout.write('\n\n' + 80*'~' + '\n')
    fout.write('a  = '+ str(coef_vec[0]) + '\n')
    fout.write('b  = '+ str(coef_vec[1]) + '\n')
    fout.write('R² = '+ str(r2))





### PLOT SECTION

# erzeuge Gradienten des Modellfehlers
grad_ERR = 200*np.gradient(ERR)

# erzeuge Modellausgabe mit optimierten Koeffizienten
U_model  = np.arange(min(U), max(U), 0.01)
dP_model = []
for u_model in U_model:
    dP_model.append(press_diff(u_model, coef_vec))


plt.style.use('ggplot')

fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(20,10))

ax_1.plot([i for i in range(len(ERR))], ERR, color='blue', label='Modellfehler')
ax_1.plot([i for i in range(len(grad_ERR))], grad_ERR, color='green', label='grad(Modellfehler)\nskaliert um Faktor 200')
ax_1.set_title('Konvergenzverhalten')
ax_1.set_xlabel('Iterationsschritt')
ax_1.set_ylabel('Modellfehler und gradient(Modelfehler)')
ax_1.legend()

ax_2.scatter(U, dP, color='blue', label='Trainingsdaten')
ax_2.plot(U_model, dP_model, color='green', label='Modell')
ax_2.set_title('Modell gegen Trainingsdaten')
ax_2.set_xlabel('Geschwindigkeit u')
ax_2.set_ylabel('Druckdifferenz dp')
ax_2.legend()

plt.legend()

plt.show()