/*============================================================================*
*                              FILE: a1.c                                     *
*                  Skeleton code: COMP10002 Assignment 1 2026                 *
*              Written by: Dr. Shaanan Cohney and Kacie Beckett               *
*        Attention Is All You Need (Single-Head Attention with KV Cache)      *
*           Edited by: Jiawei Luo.   1809823
*============================================================================*/
/*==========================================================*
*                   COMPLEXITY ANALYSIS                     *
*==========================================================*/
/*
Stage 1:O(text_len log text_len + count ^2),where count is the number
of unique tokens. The qsort call contributes O(text_len log text_len),
to sort tokens. The printing loop is two nested loops each of length of count,
contributes O(count^2), since their size relationship can't be determined
both are kept.
Stage 2:O(nd^2), the stage2 is made of three nested loops, the outer loop 
runs n times(per prompt token), the middle loop runs d times(per output component), 
the inner loop runs d times(the dot product).
Stage 3:O(n^2d), the stage3 is made of three nested lops, the outer loop runs 
n times(prompt token for i), the middle loop runs n times(prompt token for j),
the inner loop runs d times to compute the dot product. The operation is n*n*d
Stage 4:O(n^2d), the stage uses the same scorre computation as stage3,
giving O(n^2d), the softmax part adds three loops of n inside the outer i loop,
contributing 3n, which is donminated by n^2d. Totally, O(3n + n^2d)= O(n^2d)
Stage 5:O(n^2d), the stage5 is made of three nested loops, the outer loop runs n 
times(every token), the middle loop runs d times(dimention in vector), the inner loop
runs n times. The whole operation is O(n*d*n)=O(n^2d)
Stage 6:O(d^2 + (n+g)*d). In the three projections, there are two nested 
loops in each projection((d^2)*3). For the calculation of the score, computes ((n+g)*d) times,
since the cache hold a most (n+g) entries and each score is a length-d dot-product.The output
as a weighted sum of cached values is also o((n+g)*d), softmax over the values is
o(n+g), which can be ignored.

*/

/*==========================================================*
*                  PREPROCESSOR DIRECTIVES                  *
*==========================================================*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 64
#define MAX_D 64
#define MAX_GEN 32

#define NO_SOFTMAX 0
#define APPLY_SOFTMAX 1

#define MAX_TEXT_SIZE 64
#define MAX_TOKEN_LENGTH 20

/* Build the scanf format for a string reading into a fixed size char array.
   This should be used for Stage 0 when reading the prompt.*/
#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)
#define TOKEN_STR_SCANF_FORMAT "%" STR(MAX_TOKEN_LENGTH) "s "

/*==========================================================*
*                   FUNCTION PROTOTYPES                     *
*==========================================================*/

void read_input(int *n, int *d, int *g, int *text_len,
                char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1],
                int mask[MAX_TOKENS],
                double prompt[MAX_TOKENS][MAX_D],
                double gen[MAX_GEN][MAX_D],
                double wq[MAX_D][MAX_D], double wk[MAX_D][MAX_D],
                double wv[MAX_D][MAX_D]);

void create_embeddings(
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1], int text_len);

void compute_projection(int count, int d, double src[MAX_TOKENS][MAX_D],
                        double w[MAX_D][MAX_D],
                        double dest[MAX_TOKENS][MAX_D]);

void compute_attention_scores_or_weights_prompt(
    int n, int d, const int mask[MAX_TOKENS],
    double q[MAX_TOKENS][MAX_D], double k[MAX_TOKENS][MAX_D],
    double scores_or_weights[MAX_TOKENS][MAX_TOKENS],
    int apply_softmax);

void compute_attention_output_prompt(
    int n, int d, double weights[MAX_TOKENS][MAX_TOKENS],
    double v[MAX_TOKENS][MAX_D], double out[MAX_TOKENS][MAX_D]);

long compute_generation_with_cache(
    int n, int d, int g, int t, const int mask[MAX_TOKENS],
    double gen[MAX_GEN][MAX_D], double wq[MAX_D][MAX_D],
    double wk[MAX_D][MAX_D], double wv[MAX_D][MAX_D],
    double k_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double v_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double output[MAX_D]);

int compare_tokens(const void *a, const void *b);

double clamp_near_zero(double value);

void print_vector(int d, const double row[]);

void print_attention_matrix(int n,
    const double matrix[MAX_TOKENS][MAX_TOKENS]);

void print_embedding_matrix(int n, int d,
    const double matrix[MAX_TOKENS][MAX_D]);
/*==========================================================*
*                       MAIN LOOP                           *
*==========================================================*/

int main(void) {
    int n = 0;
    int d = 0;
    int g = 0;
    int text_len = 0;
    int mask[MAX_TOKENS] = {0};
    double prompt[MAX_TOKENS][MAX_D] = {{0}};
    double gen[MAX_GEN][MAX_D] = {{0}};
    double wq[MAX_D][MAX_D] = {{0}};
    double wk[MAX_D][MAX_D] = {{0}};
    double wv[MAX_D][MAX_D] = {{0}};
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1] = {{0}};

    read_input(&n, &d, &g, &text_len, embedding_table, mask, prompt, gen, wq,
               wk, wv);

    printf("Stage 1: Create Embeddings\n");
    create_embeddings(embedding_table, text_len);
    /* For stage 1 you must print the embedding table yourself.
       The later stages are done for you. */

    double q_matrix[MAX_TOKENS][MAX_D] = {{0}};
    double k_matrix[MAX_TOKENS + MAX_GEN][MAX_D] = {{0}};
    double v_matrix[MAX_TOKENS + MAX_GEN][MAX_D] = {{0}};

    printf("Stage 2: Projections\n");

    compute_projection(n, d, prompt, wq, q_matrix);
    printf("Q Projection:\n");
    print_embedding_matrix(n, d, q_matrix);

    compute_projection(n, d, prompt, wk, k_matrix);
    printf("K Projection:\n");
    print_embedding_matrix(n, d, k_matrix);

    compute_projection(n, d, prompt, wv, v_matrix);
    printf("V Projection:\n");
    print_embedding_matrix(n, d, v_matrix);

    printf("Stage 3: Attention Scores (Prompt)\n");
    double scores[MAX_TOKENS][MAX_TOKENS] = {{0}};
    compute_attention_scores_or_weights_prompt(
        n, d, mask, q_matrix, k_matrix, scores, NO_SOFTMAX);
    print_attention_matrix(n, scores);

    printf("Stage 4: Attention Weights (Prompt)\n");
    double weights[MAX_TOKENS][MAX_TOKENS] = {{0}};
    compute_attention_scores_or_weights_prompt(
        n, d, mask, q_matrix, k_matrix, weights, APPLY_SOFTMAX);
    print_attention_matrix(n, weights);

    printf("Stage 5: Attention Output (Prompt)\n");
    double out_prompt[MAX_TOKENS][MAX_D] = {{0}};
    compute_attention_output_prompt(n, d, weights, v_matrix, out_prompt);
    print_embedding_matrix(n, d, out_prompt);

    printf("Stage 6: Generated Outputs\n");
    for (int t = 0; t < g; t++) {
        double gen_output[MAX_D] = {0};
        long dot_products = compute_generation_with_cache(
            n, d, g, t, mask, gen, wq, wk, wv, k_matrix, v_matrix,
            gen_output);
        printf("Gen %d: ", t);
        print_vector(d, gen_output);
        printf("Dot products computed: %ld\n", dot_products);
    }

    return 0;
}

/*==========================================================*
*                 FUNCTIONS TO IMPLEMENT                    *
*==========================================================*/

/*satge0: read all the input needed, the Stage 1 token text, the prompt mask, the prompt
embedding rows, the generated embedding rows, and finally the three
projection matrices Wq, Wk, Wv.*/
void read_input(int *n, int *d, int *g, int *text_len,
                char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1],
                int mask[MAX_TOKENS],
                double prompt[MAX_TOKENS][MAX_D],
                double gen[MAX_GEN][MAX_D],
                double wq[MAX_D][MAX_D], double wk[MAX_D][MAX_D],
                double wv[MAX_D][MAX_D]) {
    /* Read sizes: n prompt tokens, d embedding dim, g generated tokens,
       text_len Stage 1 tokens. */
    scanf("%d", n);
    scanf("%d", d);
    scanf("%d", g);
    scanf("%d", text_len);

    //1.read tokens for stage 1
    for (int i = 0; i < *text_len; i++) {
        scanf(TOKEN_STR_SCANF_FORMAT, embedding_table[i]);
    }

    //2.read prompt mask
    for (int i = 0; i < *n; i++) {
        scanf("%d", &mask[i]);
    }

    //3.read prompt rows (n x d)
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &prompt[i][j]);
        }
    }

    //4.read generated rows (g x d)
    for (int i = 0; i < *g; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &gen[i][j]);
        }
    }

    //5.read Wq (d x d)
    for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wq[i][j]);
        }
    }

    //6.read Wk (d x d)
    for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wk[i][j]);
        }
    }

    //7.read Wv (d x d)
    for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wv[i][j]);
        }
    }
}

/*stage1:build and print the one-hot embedding table. Sorts the
token list lexicographically, removes duplicates, and prints each
unique token with a one-hot vector whose 1 sits at that token's
index in the sorted list.*/
void create_embeddings( 
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1], int text_len) {
    (void)embedding_table;
    (void)text_len;
    
    //1.use qsort to sort the token 
    qsort(embedding_table, text_len, MAX_TOKEN_LENGTH + 1, compare_tokens);
    //2.remove duplicates
    char unique_tokens[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1];
    int count = 0;
    for (int i = 0; i < text_len; i++) {
        if (i == 0 || strcmp(embedding_table[i], embedding_table[i-1]) != 0) {
            strcpy(unique_tokens[count], embedding_table[i]);
            count++;
        }
    }
       
    //3.print every unique token and their one-hot vector
    for (int i = 0; i <= count - 1; i++) {
    printf("\"%s\" -> (", unique_tokens[i]);
    for (int j = 0; j <= count - 1; j++) {
        if (j == i) {
            printf("1");
        } else {
            printf("0");
        } 
        //fill the remaining positions with space
        if (j < count - 1) {
            printf(" ");
        }
    }
        printf(")\n");
    }
}

/*Stage 2: project a set of embedding rows by a weight matrix.
Each output row is the matrix product src[i] * w, used to build
Q, K, or V depending on which weight matrix is passed in.*/
void compute_projection(int count, int d, double src[MAX_TOKENS][MAX_D],
                        double w[MAX_D][MAX_D],
                        double dest[MAX_TOKENS][MAX_D]) {
    /* Supress compiler warning for unused variables until implemented */
    (void)count;
    (void)d;
    (void)src;
    (void)w;
    (void)dest;
    
    // loop each of the count prompt tokens
    for (int i = 0; i < count; i++) {
        // loop each of the d componets
        for (int j = 0; j < d; j++) {
            //reset accumulator for each output
            double sum = 0.0;
            for (int m = 0; m < d; m++) {
                sum += src[i][m] * w[m][j]; //dot product of matric
            }
            dest[i][j] = sum;
        }
    }
}

/* Stages 3 and 4: compute either the prompt attention score matrix
   or the prompt attention weight matrix, depending on apply_softmax.
   With NO_SOFTMAX, fills scores_or_weights with scaled dot products
   QK^T / sqrt(d), masking causal (j > i) and padding positions with
   -INFINITY. With APPLY_SOFTMAX, the same masking applies and each
   row is normalised using stable softmax (rows with no unmasked
   columns are output as zeros). */
void compute_attention_scores_or_weights_prompt(
    int n, int d, const int mask[MAX_TOKENS],
    double q[MAX_TOKENS][MAX_D], double k[MAX_TOKENS][MAX_D],
    double scores_or_weights[MAX_TOKENS][MAX_TOKENS],
    int apply_softmax) {
       
    //1.point out the masked elements in score with -infinity
    //then do the calculation
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //causal mask
            if (mask[i] == 0) {
                scores_or_weights[i][j] = -INFINITY;
            } else if (mask[j] == 0) {
                scores_or_weights[i][j] = -INFINITY;
            } else if (j > i) { 
                //padding mask
                scores_or_weights[i][j] = -INFINITY;
            } else {
                //2.use the formula to calculate the score
                double sum = 0.0;
                for (int m = 0; m < d; m++) {
                    sum += q[i][m] * k[j][m];
                }
                scores_or_weights[i][j] = sum / sqrt(d);
            }
        }
        
        //softmax part
        //1.update the max score until find out the biggest one
        if (apply_softmax == APPLY_SOFTMAX) {

            double max_score = -INFINITY;
            for (int j = 0; j < n; j++) {
                if (scores_or_weights[i][j] != -INFINITY
                    && scores_or_weights[i][j] > max_score) {
                    max_score = scores_or_weights[i][j];
                }
            }

            //2.calculate the denominator of the formula
            double sum_score = 0.0;
            for (int j = 0; j < n; j++) {
                if (scores_or_weights[i][j] != -INFINITY) {
                    sum_score += exp(scores_or_weights[i][j] - max_score);
                }
            }

            //3.calculate the weights, substitute into the formula
            //the whole row turns to 0 when sum_score == 0.0
            if (sum_score == 0.0) {
                for (int j = 0; j < n; j++) {
                    scores_or_weights[i][j] = 0.0;
                }
            } else {
                //weight = 0 when score = -INFINITY
                for (int j = 0; j < n; j++) {
                    if (scores_or_weights[i][j] == -INFINITY) {
                        scores_or_weights[i][j] = 0.0;
                    } else {
                        scores_or_weights[i][j] =
                            exp(scores_or_weights[i][j] - max_score) / sum_score;
                    }
                }
            }
        }
    
    }
}  

/* Stage 5: compute the prompt attention output matrix. Each output
   row is the weighted sum of the value rows V, using the attention
   weights from Stage 4 as the blending coefficients. */
void compute_attention_output_prompt(
    int n, int d, double weights[MAX_TOKENS][MAX_TOKENS],
    double v[MAX_TOKENS][MAX_D], double out[MAX_TOKENS][MAX_D]) {
    /* Supress compiler warning for unused variables until implemented */
    (void)n;
    (void)d;
    (void)weights;
    (void)v;
    (void)out;
    /* TODO: compute the weighted sums of the value rows for the prompt. */
    //1.go through every token
    for (int i = 0; i < n; i++) {
        //2.go through every dimention in the vector
        for (int j = 0; j < d; j++) {
            //reset 
            out[i][j] = 0.0;
            //3.calculate the weighted sum
            for (int m = 0; m < n; m++) {
                out[i][j] += weights[i][m] * v[m][j];
            }

        }
    }
}

/* Stage 6: compute attention output for one generated token, reusing
   prompt K/V from the cache and appending the new K/V. Returns the
   number of dot products performed. */
long compute_generation_with_cache(
    int n, int d, int g, int t, const int mask[MAX_TOKENS],
    double gen[MAX_GEN][MAX_D], double wq[MAX_D][MAX_D],
    double wk[MAX_D][MAX_D], double wv[MAX_D][MAX_D],
    double k_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double v_cache[MAX_TOKENS + MAX_GEN][MAX_D],
    double output[MAX_D]) {
    /* Supress compiler warning for unused variables until implemented */
    (void)n;
    (void)d;
    (void)g;
    (void)t;
    (void)mask;
    (void)gen;
    (void)wq;
    (void)wk;
    (void)wv;
    (void)k_cache;
    (void)v_cache;
    (void)output;
    /* TODO: compute the next generated output using the KV cache and return
       the number of dot products performed for that generation step. */
    
    //1.multiply the embedding of the row gen[t] by the wq matrix to obtain 
    //a  vector of length d
    //the last step: count how many dot producte were computed
    long dot_count = 0;
    double new_vector[MAX_D];// set a temporary one
    for (int j = 0; j < d; j++) {
        new_vector[j] = 0.0;
        for (int m = 0; m < d; m++) {
            new_vector[j] += gen[t][m] * wq[m][j];
        }
        dot_count++;
    }
    
    //2.cpmpute K for the new tokens and append it to the cache
    for (int j = 0; j < d; j++) {
        k_cache[n+t][j] = 0.0;
        for (int m = 0; m < d; m++) {
            k_cache[n+t][j] += gen[t][m] * wk[m][j];
        }
        dot_count++;
    }
    
    //3.compute V for the new tokens and append it to the cache
    for (int j = 0; j < d; j++) {
        v_cache[n+t][j] = 0.0;
        for (int m = 0; m < d; m++) {
            v_cache[n+t][j] += gen[t][m] * wv[m][j];
        }
        dot_count++;
    }
    
    //4/calculate the scores
    double scores[MAX_TOKENS + MAX_GEN];
    //compute scaled dot product scores between q and all ached keys
    for (int i = 0; i <= n + t; i++) {
        //padding mask
        if (i < n && mask[i] == 0) {
            scores[i] = -INFINITY;
        } else {
            //like stage3,calculate the score
            double dot_product = 0.0;
            for (int m = 0; m < d; m++) {
                dot_product += new_vector[m] * k_cache[i][m];
            }
            scores[i] = dot_product / sqrt(d);
            dot_count++;
        }
    }
    
    //5. stable softmax the cached scores
    //find the biggest score
    double max_score = -INFINITY;
    for (int i = 0; i <= n + t; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    double weights[MAX_TOKENS + MAX_GEN];
    
    //calculate the deminator
    double sum_scores = 0.0;
    for (int i = 0; i <=n + t; i++) {
        if (scores[i] != -INFINITY) {
            weights[i]= exp(scores[i] - max_score);
            sum_scores += weights[i];
        } else {
            weights[i] = 0.0;
        }
    }
    
    //deminator can't be zero
    if (sum_scores == 0.0) {
        for (int i = 0; i <=n + t; i++) {
            weights[i] = 0.0;
        } 
    } else {
        for (int i = 0; i <=n + t; i++) {
            weights[i] = weights[i] / sum_scores;
        }
    }
    
    //6.weighted sum of cached value vectirs
    for (int j = 0; j < d; j++) { // for every dimention
        output[j] = 0.0;
        //loop the cached value
        for (int i = 0; i <=n + t; i++) {
            output[j] += weights[i] * v_cache[i][j];
        }
        dot_count++;
    }
    return dot_count;
}

/*==========================================================*
*                    HELPER FUNCTIONS                       *
*==========================================================*/

/* Optional helper for qsort based token sorting */
int compare_tokens(const void *a, const void *b) {
    return strcmp((const char *)a, (const char *)b);
}

/* Set small doubles to 0.0 to remove floating point noise */
double clamp_near_zero(double value) {
    if (fabs(value) < 0.0005) {
        return 0.0;
    }
    return value;
}

/* print an n by n attention matrix with each row on a new line */
void print_attention_matrix(int n,
    const double matrix[MAX_TOKENS][MAX_TOKENS]) {
    for (int i = 0; i < n; i++) {
        print_vector(n, matrix[i]);
    }
}

/* print an n by d embedding matrix with each row on a new line */
void print_embedding_matrix(int n, int d,
    const double matrix[MAX_TOKENS][MAX_D]) {
    for (int i = 0; i < n; i++) {
        print_vector(d, matrix[i]);
    }
}

/* print a vector on a new row with each value seperated by a space, and small
   values clamped to 0.0*/
void print_vector(int d, const double row[]) {
    for (int i = 0; i < d; i++) {
        double value = clamp_near_zero(row[i]);
        if (i == d - 1) {
            printf("%.3f", value);
        } else {
            printf("%.3f ", value);
        }
    }
    printf("\n");
}