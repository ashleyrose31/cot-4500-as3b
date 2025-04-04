#Question 1
import numpy as np

# Define the coefficient matrix and constants
A = np.array([[2, -1, 1],
              [1, 3, 1],
              [-1, 5, 4]], dtype=float)

b = np.array([6, 0, -3], dtype=float)

solution = np.linalg.solve(A, b)

print(solution)

#Question 2 
import numpy as np

def lu_factorization(matrix):
   
    n = len(matrix)
    L = np.eye(n)
    U = matrix.copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]  
            L[j, i] = factor            
            U[j, i:] -= factor * U[i, i:]  

    return L, U
#Question 2a
def determinant(matrix):
    
    return np.linalg.det(matrix)


matrix = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)


L, U = lu_factorization(matrix)


det = determinant(matrix)

#Question 2a
print("Determinant:", det)

#Question 2b
print("\nL Matrix:")
print(L)

#Question 2c
print("\nU Matrix:")
print(U)


#Question 3 - Determining if the matrix is diagonally dominate 
B = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        row = matrix[i]
        diagonal = abs(row[i])
        off_diagonal_sum = sum(abs(row[j]) for j in range(n) if j != i)
        if diagonal < off_diagonal_sum:
            return False
    return True


print("\nIs the matrix positive definite?", is_diagonally_dominant(B))

#Question 4 - Determine if the matrix is a positive definite

C = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

eigenvalues = np.linalg.eigvals(C)
is_positive_definite = np.all(eigenvalues > 0)

print("\nIs the matrix positive definite?", is_positive_definite)