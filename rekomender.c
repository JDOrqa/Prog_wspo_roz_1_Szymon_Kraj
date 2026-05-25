#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

#define MAX_USERS    10000
#define MAX_PRODS    1000
#define TOP_K        50
#define EPSILON      1e-12

typedef struct {
    int prod_id;
    double rating;
} RatingByUser;

typedef struct {
    int user_id;
    double rating;
} RatingByProduct;

RatingByUser *user_items[MAX_USERS];
int user_items_cnt[MAX_USERS];

RatingByProduct *prod_users[MAX_PRODS];
int prod_users_cnt[MAX_PRODS];

double norms[MAX_USERS];


int cmp_user_by_prod(const void *a, const void *b) {
    return ((RatingByUser*)a)->prod_id - ((RatingByUser*)b)->prod_id;
}

int load_ratings(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;

    for (int i = 0; i < MAX_USERS; i++) user_items_cnt[i] = 0;
    for (int p = 0; p < MAX_PRODS; p++) prod_users_cnt[p] = 0;

    int uid, pid, total = 0;
    double rate;
    int *user_temp = calloc(MAX_USERS, sizeof(int));
    int *prod_temp = calloc(MAX_PRODS, sizeof(int));
    if (!user_temp || !prod_temp) { fclose(f); return 0; }

    // 1. przebieg – zliczanie
    while (fscanf(f, "%d,%d,%lf", &uid, &pid, &rate) == 3) {
        if (uid < 1 || uid > MAX_USERS || pid < 1 || pid > MAX_PRODS) continue;
        user_temp[uid-1]++;
        prod_temp[pid-1]++;
        total++;
    }
    rewind(f);

    // Alokacja
    for (int i = 0; i < MAX_USERS; i++)
        if (user_temp[i])
            user_items[i] = malloc(user_temp[i] * sizeof(RatingByUser));
    for (int p = 0; p < MAX_PRODS; p++)
        if (prod_temp[p])
            prod_users[p] = malloc(prod_temp[p] * sizeof(RatingByProduct));

    memset(user_temp, 0, MAX_USERS * sizeof(int));
    memset(prod_temp, 0, MAX_PRODS * sizeof(int));

    // 2. przebieg – wypełnienie
    while (fscanf(f, "%d,%d,%lf", &uid, &pid, &rate) == 3) {
        if (uid < 1 || uid > MAX_USERS || pid < 1 || pid > MAX_PRODS) continue;
        int u = uid-1, p = pid-1;

        int idx_u = user_temp[u]++;
        user_items[u][idx_u].prod_id = p;
        user_items[u][idx_u].rating = rate;
        user_items_cnt[u]++;

        int idx_p = prod_temp[p]++;
        prod_users[p][idx_p].user_id = u;
        prod_users[p][idx_p].rating = rate;
        prod_users_cnt[p]++;
    }

    fclose(f);
    free(user_temp);
    free(prod_temp);

    // Sortowanie list użytkowników
    for (int i = 0; i < MAX_USERS; i++)
        if (user_items_cnt[i] > 0)
            qsort(user_items[i], user_items_cnt[i], sizeof(RatingByUser), cmp_user_by_prod);

    printf("Wczytano %d ocen.\n", total);
    return 1;
}


void compute_norms() {
    for (int u = 0; u < MAX_USERS; u++) {
        double sum = 0.0;
        for (int i = 0; i < user_items_cnt[u]; i++) {
            double r = user_items[u][i].rating;
            sum += r * r;
        }
        norms[u] = sqrt(sum);
    }
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Użycie: %s <plik_ocen.csv>\n", argv[0]);
        return 1;
    }
    if (!load_ratings(argv[1])) {
        fprintf(stderr, "Błąd wczytywania.\n");
        return 1;
    }
    compute_norms();

    printf("Pierwsze 10 norm wektorow ocen:\n");
for (int i = 0; i < 10 && i < MAX_USERS; i++) {
    printf("Uzytkownik %d: norma = %.4f\n", i+1, norms[i]);
}

    
    for (int i = 0; i < MAX_USERS; i++) free(user_items[i]);
    for (int p = 0; p < MAX_PRODS; p++) free(prod_users[p]);
    return 0;
}
