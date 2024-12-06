% Enfermedades
enfermedad(gripe).
enfermedad(rubeola).
enfermedad(malaria).
enfermedad(hepatitis).
enfermedad(tuberculosis).
enfermedad(anemia).
% S�ntomas, seg�n enfermedad
% Posibles consultas:
% Mostrar una enfermedad y sus s�ntomas o las enfermedades y s�ntomas
% Determinar un s�ntoma a que enfermedad(es) pertenece
sintomade(tos, gripe). % La tos es s�ntoma de gripe
sintomade(cansancio, gripe).
sintomade(fiebre, gripe).
sintomade(dolorcabeza, gripe).
%COMPLETAR: Continuar con el resto s�ntomas de las enfermedades del
cuadro
% Reglas
% Probabilidad una persona puede tener una enfermedad X dado N s�ntomas
% La funci�n buscar busca las enfermedades que contengan como m�nimo los s�ntomas
suministrados: 1er par�metro (lista de enfermedades), 2do par�metro (Enfermedad), 3er par�metro
cantidad de ocurrencias
buscar([], E , 0). % lista vac�a en caso que no haya coincidencias
buscar(X , E , 1) :- sintomade(X, E). % un s�ntoma para enfermedad (no hay lista)
% lista de sintomas (primero, restantes), verifica la E, total coincidencias
buscar([X|Xs] , E , P) :- enfermedad(E) , buscar(X , E , S1) , buscar(Xs , E ,S2) , P is S1 + S2.
% Funci�n que devuelve la cantidad de s�ntomas totales de la enfermedad seleccionada
cantSint(E , C) :- findall(X , sintoma(X, E) , L) , length(L , R), C is R.
% Esta funci�n es parecida a la de buscar con la excepci�n que en el �ltimo argumento entrega el
porcentaje de probabilidad de la enfermedad
diagnostico([X|Xs] , E , K) :- buscar([X|Xs] , E , P) , cantSint(E , T) , K is P * 100 / T.
% Medicina dependiendo la enfermedad
medicinade(contrex, gripe). %contrex es medicamento de la gripe
medicinade(jarabe, gripe).
medicinade(pastillas, tubercolosis).
medicinade(vacuna, malaria).
%App. Basadas en el Conocimiento
?- medicinade(X, malaria).
X = vacuna;
?- recetade(M, tos).
M = contrex;
M = jarabe;
?- especialistade(X, gripe).
X = otorrino;
?- mereceta(E, M, gripe).
E = otorrino,
M = contrex ;
E = otorrino,
M = jarabe;
% Reglas
% receta m�dica seg�n s�ntoma
recetade(M, S):-sintomade(S, Z),medicinade(M, Z).
especialistade(otorrino, gripe).
especialistade(nutricionista, anemia).
especialistade(endocrinologia, hepatitis).
especialistade(medicinageneral, rub�ola).
especialistade(nutricionista, tubercolosis).
especialistade(medicinageneral, malaria).
% Reglas
% Especialista seg�n la enfermedad y medicina a recetar
atiende_especialista(E, S):- sintoma_de(S,Z),especialista_de(E, Z).
mereceta(Es, M, E):-medicinade(M, E),sintomade(S, E), atiendeespecialista(Es,S).
