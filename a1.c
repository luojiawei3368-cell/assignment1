/*============================================================================*
*                              FILE: a1.c                                     *
*                  Skeleton code: COMP10002 Assignment 1 2026                 *
*              Written by: Dr. Shaanan Cohney and Kacie Beckett               *
*        Attention Is All You Need (Single-Head Attention with KV Cache)      *
*           Edited by: [PLEASE ADD YOUR FULL NAME AND STUDENT ID HERE]        *
*============================================================================*/
/*==========================================================*
*                   COMPLEXITY ANALYSIS                     *
*==========================================================*/
/*
Stage 1:
Stage 2:
Stage 3:
Stage 4:
Stage 5:
Stage 6:
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

void read_input(int *n, int *d, int *g, int *text_len,
                char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1],
                int mask[MAX_TOKENS],
                double prompt[MAX_TOKENS][MAX_D],
                double gen[MAX_GEN][MAX_D],
                double wq[MAX_D][MAX_D], double wk[MAX_D][MAX_D],
                double wv[MAX_D][MAX_D]) {
    /* Supress compiler warning for unused variables until implemented */
    
    /* TODO: read all the input from stdin. For reading tokens into the fixed size
       char array use the constant TOKEN_STR_SCANF_FORMAT with scanf which sets the
       maximum number of chars that can be read. */
        scanf("%d", n);
       scanf("%d", d);
       scanf("%d", g);
       scanf("%d", text_len);
       //2.read tokens
       for (int i = 0; i < *text_len; i++) {
        scanf(TOKEN_STR_SCANF_FORMAT, embedding_table[i]);
       }
       //3.read mask
       for (int i = 0; i < *n; i++) {
        scanf("%d", &mask[i]);
       }
       //4.read prompt (nxd)
       for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &prompt[i][j]);
        }
       }
       //5.read gen (gxd)
       for (int i = 0; i < *g; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &gen[i][j]);
        }
       }
       //6.read wq (dxd)
       for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wq[i][j]);
        }
       }
       //7.read wk(dxd)
       for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wk[i][j]);
        }
       }
       //8.read wv(dxd)
       for (int i = 0; i < *d; i++) {
        for (int j = 0; j < *d; j++) {
            scanf("%lf", &wv[i][j]);
        }
       }
}

void create_embeddings(
    char embedding_table[MAX_TEXT_SIZE][MAX_TOKEN_LENGTH + 1], int text_len) {
    (void)embedding_table;
    (void)text_len;
    /* TODO: collect the unique tokens, sort them lexicographically,
       and print the one-hot vectors for the unique tokens. */
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
            if (j < count - 1) {
                printf(" ");
            }
        }
        printf(")\n");
       }
}

void compute_projection(int count, int d, double src[MAX_TOKENS][MAX_D],
                        double w[MAX_D][MAX_D],
                        double dest[MAX_TOKENS][MAX_D]) {
    /* Supress compiler warning for unused variables until implemented */
    (void)count;
    (void)d;
    (void)src;
    (void)w;
    (void)dest;
    /* TODO: multiply each source row by the projection matrix. */
}

void compute_attention_scores_or_weights_prompt(
    int n, int d, const int mask[MAX_TOKENS],
    double q[MAX_TOKENS][MAX_D], double k[MAX_TOKENS][MAX_D],
    double scores_or_weights[MAX_TOKENS][MAX_TOKENS],
    int apply_softmax) {
    /* Supress compiler warning for unused variables until implemented */
    (void)n;
    (void)d;
    (void)mask;
    (void)q;
    (void)k;
    (void)scores_or_weights;
    (void)apply_softmax;
    /* TODO: compute scaled dot-product attention scores with causal and
       padding masking. If apply_softmax == APPLY_SOFTMAX, convert each row
       into attention weights with a stable softmax. */
}

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
}

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
    return 0;
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