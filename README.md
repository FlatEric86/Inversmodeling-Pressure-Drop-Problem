# Inversmodeling-Pressure-Drop-Problem
Ein auf das Gradientenabstiegsverfahren basiertes Tool zum Fitten einer Polynomfunktion 2. Grades (semi-analytisch).
Natürlich gibt es wesentlich effizienter arbeitende Fitting-Funktionen in Bibliotheken wie ScikitLearn oder SciPy, oder man hätte zumindest für die Optimierung TensorFlow oder Dergleichen einsetzen können, jedoch wollte ich die Gelegenheit nutzen, einmal selber die Gradientenabstiegmethode zu implementieren, da ich diese zum Zeitpunkt des Projektes im Studium gelernt hatte. 
Aufgrund der geringen Anzahl an Trainingsdaten wurde keine Modellvalidierung anhand von seperaten Testdaten/Validierungsdaten unternommen. 

Semi-analytisch mein in diesem Fall, dass die Ableitung der Fehlerfunktion zur Optimierung der Koeffizienten nicht numerisch aproximiert wurde (Differenzenquotient), sondern analytisch also als Differentialquotient in das Modell implementiert wurde. Somit kann das Modell auch nur eine Polynomfunktion 2. Grades fitten, dafür jedoch mit sehr hoher Genauigkeit.
