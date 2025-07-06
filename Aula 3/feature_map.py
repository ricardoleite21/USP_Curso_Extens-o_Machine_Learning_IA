from math import comb
#propósito: produzir combinações de polinômios X1, X2, .. Xn para poder 
#efetuar o feature mapping e utilizar polynomial regression pra curve fitting;

#essa função serve para contornar as chatices do numpy;
def Length(input):
    try:
        num = len(input)
        return num
    except:
        return 1
    
def Max(array):
    max = array[0]
    for i in range(Length(array)):
        if (array[i] > max):
            max = array[i]
    return max

def Min(array):
    min = array[0]
    for i in range(Length(array)):
        if (array[i] < min):
            min = array[i]
    return min

def Combination(polynomials, number, combinations, end, start, size_comb, i, n):
    #casos bases da recursão
    if (i == size_comb):
        combinations[n[0]] = number
        n[0] += 1 
        return

    if (start > end):
        return

    save = number

    if (Length(polynomials) == 1):
        number = number * polynomials
    else:     
        number = number * polynomials[start]

    #mover a fila +1;
    Combination(polynomials, number, combinations, end, start, size_comb, i + 1, n) 
    #mover os números +1;
    Combination(polynomials, save, combinations, end, start + 1, size_comb, i, n)

#para calcular o tamanho do vetor de fmapping;
def Combinations_Replacement (n, r):
    i = 1
    result = 0

    while i <= r:
        result += comb(n + i - 1, i)
        i += 1

    return result

#assumo que a ordem dos polinômios não importe (não vejo o por que);
#para uma sequencia [X1 .. XN] de N polinomios, retorna uma lista de 
#suas combinações;
def Polynomial_Regression(polynomials, size_pol, size_comb):
    if (Length(polynomials) == 0): 
        return 0
    
    size = Combinations_Replacement (size_pol, size_comb)

    mapping = [0] * (size + 1)
    
    mapping[0] = 1

    n = [1]

    index = 1
    while index <= size_comb:
        number = 1
        Combination (polynomials, number, mapping, size_pol - 1, 0, index, 0, n)
        index += 1

    return mapping

    
                                        