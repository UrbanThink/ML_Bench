# MobiShift
## 1 - Bien démarrer
### Installer python3
Lien vers un [tutoriel](https://docs.python-guide.org/starting/install3/linux/)
### Comment installer les dépendances

Pour faire fonctionner ces scripts, vous aurez besoin d'un certains nombres de modules. 
Pour installer un module manquant, signalé par le message d'erreur suivant : `ModuleNotFoundError: No module named '[nom du package]'`.

Par exemple : 

`ModuleNotFoundError: No module named 'fpdf'` 
est résolu par la commande
`python3 -m pip install fpdf`

Les commandes suivantes ont été executés :
```
python3 -m pip install xlsxwriter
python3 -m pip install icecream
python3 -m pip install termcolor
python3 -m pip install fpdf
```
