#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-6
#define MAX_ITER 10000

// Build FDM matrix A and RHS vector b for Laplace equation
void findFDM(double dx, double ***A, double **b, int *n_out){
    int N_points = (int)round(1.0/dx) + 1;
    int n = N_points - 2;    // interior points per row
    *n_out = n;
    int size = n * n;         // total unknowns

    // Allocate matrix A
    *A = (double **)calloc(size, sizeof(double *));
    for(int i = 0; i < size; i++)
        (*A)[i] = (double *)calloc(size, sizeof(double));

    // Fill A using 5-point stencil (row-major flattening)
    for(int row = 0; row < size; row++){
        (*A)[row][row] = 4.0;

        // left neighbor
        if(row % n != 0) (*A)[row][row - 1] = -1.0;
        // right neighbor
        if((row+1) % n != 0) (*A)[row][row + 1] = -1.0;
        // top neighbor
        if(row >= n) (*A)[row][row - n] = -1.0;
        // bottom neighbor
        if(row < size - n) (*A)[row][row + n] = -1.0;
    }

    // Allocate RHS vector b
    *b = (double *)calloc(size, sizeof(double));

    // Apply top boundary (T=1)
    for(int ix = 0; ix < n; ix++){
        int row = (n-1)*n + ix;  // last interior row
        (*b)[row] = 1.0;
    }

    printf("Matrix A and vector b constructed for dx=%.3f, n=%d, size=%d\n", dx, n, size);
}

// Dump matrix A to txt file for MATLAB
void export_matrix_to_txt(double **A, int size, const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error opening file for writing.\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(fp, "%.10f%c", A[i][j], (j == size-1 ? '\n' : ' '));
        }
    }
    fclose(fp);
    printf("Matrix A written to %s\n", filename);
}


// SOR for solving Ax=b
void sor_solver(double **A, double *b, double w, int size, double e){
    double *x0 = calloc(size, sizeof(double));
    double *x1 = calloc(size, sizeof(double));
    // initial guess -> all zeroes 
    for(int i=0; i<size; i++){
        x0[i] = 0; // old x
        x1[i] = 0; // updated x
    }
    double max_error, error;
    int no_iter=0;
   
    do{
        // one iteration step
        max_error = 0; 
        error = 0;
        for(int i=0; i<size; i++){
            double sigma1=0, sigma2=0;

            // sigma calculations
            for(int j=0; j<size; j++){
                if(j<i) sigma1 += A[i][j]*x1[j];
                if(j>i) sigma2 += A[i][j]*x0[j]; 
            }
            
            // x updation step
            x1[i] = (b[i]-sigma1-sigma2)/A[i][i]; 
            x1[i] = x0[i] + w*(x1[i]-x0[i]);

            // error calculation
            error = fabs(x1[i]-x0[i]);
            if(max_error < error) max_error = error;
        }

        // update x0 with x1
        for(int i=0; i<size; i++){
            x0[i] = x1[i];
        }

        // iteration number update
        no_iter++;
    } while(max_error > e);

    printf("SOR converged in %d iterations\n", no_iter);
    // for(int i=0; i<size; i++){
    //     printf("x[%d] = %lf\n", i, x1[i]);
    // }
}

// Steepest Descent for solving Ax=b
void matvec(double **A, double *x, double *prod, int size){
    for(int i = 0; i < size; i++){
        prod[i] = 0.0;
        for(int j = 0; j < size; j++){
            prod[i] += A[i][j] * x[j];
        }
    }
}

double dot(double *v1, double *v2, int size){
    double sum = 0.0;
    for(int i = 0; i < size; i++)
        sum += v1[i] * v2[i];
    return sum;
}

void steepest_descent(double **A, double *b, int size, double e){
    double *x = calloc(size, sizeof(double));
    double *r = calloc(size, sizeof(double));
    double *Ar = calloc(size, sizeof(double));
    
    // initial guess -> all zeroes 
    for(int i=0; i<size; i++){
        x[i] = 0; // x
    }
    
    int no_iter = 0; 
    double max_res; 

    // initial residual -> r = b-Ax
    matvec(A, x, Ar, size);
    for(int i = 0; i < size; i++){
        r[i] = b[i] - Ar[i];
    }

    do{
        // compute Ar = A*r
        matvec(A, r, Ar, size);

        // step size alpha = (r^T r) / (r^T A r)
        double alpha = dot(r, r, size) / dot(r, Ar, size);

        // update x = x + alpha * r
        for(int i = 0; i < size; i++)
            x[i] += alpha * r[i];

        // update residual r = r - alpha * A*r
        for(int i = 0; i < size; i++)
            r[i] -= alpha * Ar[i];

        // max residual for convergence
        max_res = 0.0;
        for(int i = 0; i < size; i++){
            if(fabs(r[i]) > max_res) max_res = fabs(r[i]);
        }

        no_iter++;
    } while(max_res > e && no_iter < MAX_ITER);

    printf("Steepest Descent converged in %d iterations\n", no_iter);
    // for(int i = 0; i < size; i++){
    //     printf("x[%d] = %lf\n", i, x[i]);
    // }

    free(x);
    free(r);
    free(Ar);
}

// Minimum Residual for solving Ax=b
void minimum_residual(double **A, double *b, int size, double e){
    double *x = calloc(size, sizeof(double));
    double *r = calloc(size, sizeof(double));
    double *Ar = calloc(size, sizeof(double));
    
    // initial guess -> all zeroes 
    for(int i=0; i<size; i++){
        x[i] = 0; // x
    }
    
    int no_iter = 0; 
    double max_res; 

    // initial residual -> r = b-Ax
    matvec(A, x, Ar, size);
    for(int i = 0; i < size; i++){
        r[i] = b[i] - Ar[i];
    }

    do{
        // compute Ar = A*r
        matvec(A, r, Ar, size);

        // step size alpha = (r^T r) / (r^T A r)
        double alpha = dot(Ar, r, size) / dot(Ar, Ar, size);

        // update x = x + alpha * r
        for(int i = 0; i < size; i++)
            x[i] += alpha * r[i];

        // update residual r = r - alpha * A*r
        for(int i = 0; i < size; i++)
            r[i] -= alpha * Ar[i];

        // max residual for convergence
        max_res = 0.0;
        for(int i = 0; i < size; i++){
            if(fabs(r[i]) > max_res) max_res = fabs(r[i]);
        }

        no_iter++;
    } while(max_res > e && no_iter < MAX_ITER);

    printf("Minimum Residual converged in %d iterations\n", no_iter);
    // for(int i = 0; i < size; i++){
    //     printf("x[%d] = %lf\n", i, x[i]);
    // }

    free(x);
    free(r);
    free(Ar);
}

// Conjugate Gradient for solving Ax=b
void conjugate_gradient(double **A, double *b, int size, double tol) {
    double *x = calloc(size, sizeof(double));
    double *r = calloc(size, sizeof(double));
    double *p = calloc(size, sizeof(double));
    double *Ap = calloc(size, sizeof(double));

    // initial guess x = 0
    for(int i = 0; i < size; i++) {
        x[i] = 0.0;
    }

    // initial residual r = b - A*x = b
    for(int i = 0; i < size; i++)
        r[i] = b[i];

    // initial search direction p = r
    for(int i = 0; i < size; i++)
        p[i] = r[i];

    int iter = 0;
    double max_res;

    do {
        // compute Ap = A*p
        matvec(A, p, Ap, size);

        // alpha = (r^T r) / (p^T Ap)
        double alpha = dot(r, r, size) / dot(p, Ap, size);

        // update x = x + alpha * p
        for(int i = 0; i < size; i++)
            x[i] += alpha * p[i];

        // update residual r_new = r - alpha * Ap
        double *r_new = calloc(size, sizeof(double));
        for(int i = 0; i < size; i++)
            r_new[i] = r[i] - alpha * Ap[i];

        // beta = (r_new^T r_new) / (r^T r)
        double beta = dot(r_new, r_new, size) / dot(r, r, size);

        // update search direction p = r_new + beta * p
        for(int i = 0; i < size; i++)
            p[i] = r_new[i] + beta * p[i];

        // update residual r = r_new
        for(int i = 0; i < size; i++)
            r[i] = r_new[i];

        // compute max residual for convergence
        max_res = 0.0;
        for(int i = 0; i < size; i++){
            if(fabs(r[i]) > max_res) max_res = fabs(r[i]);
        }

        free(r_new);
        iter++;
    } while(max_res > tol && iter < MAX_ITER);

    printf("Conjugate Gradient converged in %d iterations\n", iter);
    // for(int i = 0; i < size; i++){
    //     printf("x[%d] = %lf\n", i, x[i]);
    // }

    free(x);
    free(r);
    free(p);
    free(Ap);
}

// BiCGSTAB for solving Ax=b
void bicgstab(double **A, double *b, int size, double tol) {
    double *x = calloc(size, sizeof(double)); // initial guess x0=0
    double *r = calloc(size, sizeof(double));
    double *r_hat = calloc(size, sizeof(double));
    double *p = calloc(size, sizeof(double));
    double *v = calloc(size, sizeof(double));
    double *s = calloc(size, sizeof(double));
    double *t = calloc(size, sizeof(double));

    // initial residual r0 = b - A*x0 = b
    for(int i=0; i<size; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        r_hat[i] = r[i]; // choose shadow residual
        p[i] = 0.0;
        v[i] = 0.0;
    }

    double rho_old = 1.0, alpha = 1.0, omega_old = 1.0;
    double rho_new, beta, omega;
    int iter = 0;
    double max_res;

    do {
        rho_new = dot(r_hat, r, size);
        if(fabs(rho_new) < 1e-30) {
            printf("Breakdown: rho ~ 0\n");
            break;
        }

        if(iter == 0) {
            for(int i=0; i<size; i++) p[i] = r[i];
        } else {
            beta = (rho_new / rho_old) * (alpha / omega_old);
            for(int i=0; i<size; i++)
                p[i] = r[i] + beta * (p[i] - omega_old * v[i]);
        }

        // v = A*p
        matvec(A, p, v, size);

        alpha = rho_new / dot(r_hat, v, size);

        // s = r - alpha * v
        for(int i=0; i<size; i++)
            s[i] = r[i] - alpha * v[i];

        // check small residual
        max_res = 0.0;
        for(int i=0; i<size; i++)
            if(fabs(s[i]) > max_res) max_res = fabs(s[i]);
        if(max_res < tol) {
            for(int i=0; i<size; i++)
                x[i] += alpha * p[i];
            break;
        }

        // t = A*s
        matvec(A, s, t, size);

        omega = dot(t, s, size) / dot(t, t, size);

        // update x = x + alpha*p + omega*s
        for(int i=0; i<size; i++)
            x[i] += alpha * p[i] + omega * s[i];

        // update r = s - omega*t
        for(int i=0; i<size; i++)
            r[i] = s[i] - omega * t[i];

        rho_old = rho_new;
        omega_old = omega;
        iter++;

        // max residual for convergence
        max_res = 0.0;
        for(int i=0; i<size; i++)
            if(fabs(r[i]) > max_res) max_res = fabs(r[i]);

    } while(max_res > tol && iter < MAX_ITER);

    printf("BICGSTAB converged in %d iterations\n", iter);
    // for(int i=0; i<size; i++)
    //     printf("x[%d] = %lf\n", i, x[i]);

    free(x); 
    free(r); 
    free(r_hat); 
    free(p); 
    free(v); 
    free(s); 
    free(t);
}




int main(){
    double dx = 0.08;    // grid spacing
    double **A = NULL;
    double *b = NULL;
    int n;

    findFDM(dx, &A, &b, &n);
    int size = n * n;

    // Print entire matrix A
    // printf("\nMatrix A:\n");
    // for(int i = 0; i < size; i++){
    //     for(int j = 0; j < size; j++){
    //         printf("%6.1f ", A[i][j]);
    //     }
    //     printf("\n");
    // }

    // Print entire vector b
    // printf("\nVector b:\n");
    // for(int i = 0; i < size; i++){
    //     printf("%6.3f\n", b[i]);
    // }

    clock_t start, end;
    double cpu_time_used;

    start = clock();

     //double omega = 1.6017;
     //sor_solver(A, b, omega, size, TOL);
    //steepest_descent(A, b, size, TOL);
     //minimum_residual(A, b, size, TOL);
     //conjugate_gradient(A, b, size, TOL);
    bicgstab(A, b, size, TOL);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Solver took %f seconds.\n", cpu_time_used);

    export_matrix_to_txt(A, size, "fdm_matrix.txt");


    

    // Free memory
    for(int i = 0; i < size; i++) free(A[i]);
    free(A);
    free(b);

    return 0;
}
