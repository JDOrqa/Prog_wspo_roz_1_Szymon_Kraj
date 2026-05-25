#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Stałe – rozmiary dla docelowej skali
 * ---------------------------------------------------------------------- */
#define MAX_USERS    10000
#define MAX_PRODUCTS 1000
#define MAX_LINE     256        
#define TOP_K        5

/* -------------------------------------------------------------------------
 * Dane globalne (wypełniane podczas wczytywania)
 * ---------------------------------------------------------------------- */
static float ratings[MAX_USERS][MAX_PRODUCTS];   /* macierz ocen (rzadka)   */
static float sim_par[MAX_USERS][MAX_USERS];      /* wynik równoległy        */
static float sim_seq[MAX_USERS][MAX_USERS];      /* wynik sekwencyjny       */
static float norm[MAX_USERS];                    /* pre‑obliczone normy     */

static int num_users    = MAX_USERS;
static int num_products = MAX_PRODUCTS;

/* -------------------------------------------------------------------------
 * Struktury dla wątków
 * ---------------------------------------------------------------------- */
typedef struct {
    int start_row;      /* pierwszy wiersz obsługiwany przez wątek */
    int end_row;        /* ostatni wiersz (wyłączny)                */
    int thread_id;
    float (*out)[MAX_USERS]; /* wskaźnik do tablicy wyjściowej      */
} ThreadArgs;

typedef struct {
    int   product;
    float score;
} Rec;

/* =========================================================================
 * PODOBIEŃSTWO COSINUSOWE – wersja z pre‑obliczonymi normami
 * ====================================================================== */
static inline float cosine_similarity(int u, int v) {
    float dot = 0.0f;
    for (int p = 0; p < num_products; p++)
        dot += ratings[u][p] * ratings[v][p];
    if (norm[u] < 1e-9f || norm[v] < 1e-9f) return 0.0f;
    return dot / (norm[u] * norm[v]);
}

/* -------------------------------------------------------------------------
 * Obliczenie norm wektorów ocen (długość euklidesowa)
 * ---------------------------------------------------------------------- */
static void compute_norms(void) {
    for (int u = 0; u < num_users; u++) {
        float sum = 0.0f;
        for (int p = 0; p < num_products; p++)
            sum += ratings[u][p] * ratings[u][p];
        norm[u] = sqrtf(sum);
    }
}

/* =========================================================================
 * FUNKCJA WĄTKU – oblicza podobieństwo dla przydzielonych wierszy
 * ====================================================================== */
static void *thread_func(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    for (int i = a->start_row; i < a->end_row; i++)
        for (int j = 0; j < num_users; j++)
            a->out[i][j] = cosine_similarity(i, j);
    return NULL;
}

/* =========================================================================
 * BUDOWANIE MACIERZY PODOBIEŃSTWA – równolegle
 * ====================================================================== */
static double build_similarity_parallel(int n_threads,
                                        float out[MAX_USERS][MAX_USERS]) {
    memset(out, 0, sizeof(float) * MAX_USERS * MAX_USERS);

    pthread_t   *threads = malloc((size_t)n_threads * sizeof(pthread_t));
    ThreadArgs  *args    = malloc((size_t)n_threads * sizeof(ThreadArgs));
    if (!threads || !args) { perror("malloc"); exit(1); }

    int base  = num_users / n_threads;
    int extra = num_users % n_threads;
    int row   = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int t = 0; t < n_threads; t++) {
        int load = base + (t < extra ? 1 : 0);
        args[t].thread_id = t;
        args[t].start_row = row;
        args[t].end_row   = row + load;
        args[t].out       = out;
        row += load;
        pthread_create(&threads[t], NULL, thread_func, &args[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    free(threads);
    free(args);

    return (t_end.tv_sec - t_start.tv_sec) +
           (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
}

/* =========================================================================
 * BUDOWANIE MACIERZY PODOBIEŃSTWA – sekwencyjnie
 * ====================================================================== */
static double build_similarity_sequential(float out[MAX_USERS][MAX_USERS]) {
    memset(out, 0, sizeof(float) * MAX_USERS * MAX_USERS);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int i = 0; i < num_users; i++)
        for (int j = 0; j < num_users; j++)
            out[i][j] = cosine_similarity(i, j);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec - t_start.tv_sec) +
           (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
}

/* =========================================================================
 * WERYFIKACJA – porównanie całych macierzy (dla pełnego zakresu)
 * (dla 10000×10000 trwa to chwilę, ale tylko raz)
 * ====================================================================== */
static void verify_results(float par[MAX_USERS][MAX_USERS],
                           float seq[MAX_USERS][MAX_USERS]) {
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (int i = 0; i < num_users; i++)
        for (int j = 0; j < num_users; j++) {
            float e = fabsf(par[i][j] - seq[i][j]);
            if (e > max_err) max_err = e;
            sum_err += e;
        }
    float avg_err = sum_err / (float)(num_users * num_users);
    printf("  Max blad:  %.2e\n", max_err);
    printf("  Avg blad:  %.2e\n", avg_err);
    printf("  Status:    %s\n", max_err < 1e-5f ? "OK (wyniki zgodne)" : "BLAD!");
}

/* =========================================================================
 * REKOMENDACJE – user‑based collaborative filtering
 * ====================================================================== */
static void recommend_top_k(int target_user,
                             float sim[MAX_USERS][MAX_USERS]) {
    Rec recs[MAX_PRODUCTS];
    int n_recs = 0;

    for (int p = 0; p < num_products; p++) {
        if (ratings[target_user][p] > 0.0f) continue;   /* już oceniony */

        float weighted = 0.0f;
        float sim_sum  = 0.0f;

        for (int v = 0; v < num_users; v++) {
            if (v == target_user) continue;
            if (ratings[v][p] < 1e-9f) continue;        /* nie ocenił p */

            float s = sim[target_user][v];
            if (s > 0.0f) {
                weighted += s * ratings[v][p];
                sim_sum  += s;
            }
        }

        if (sim_sum > 1e-9f) {
            recs[n_recs].product = p;
            recs[n_recs].score   = weighted / sim_sum;
            n_recs++;
        }
    }

    /* sortowanie bąbelkowe malejąco */
    for (int i = 0; i < n_recs - 1; i++)
        for (int j = i + 1; j < n_recs; j++)
            if (recs[j].score > recs[i].score) {
                Rec tmp = recs[i];
                recs[i] = recs[j];
                recs[j] = tmp;
            }

    int show = n_recs < TOP_K ? n_recs : TOP_K;
    printf("\n  %-12s  %-22s\n", "Produkt", "Przewidywana ocena");
    printf("  %-12s  %-22s\n", "------------", "--------------------");
    if (show == 0) {
        printf("  (brak rekomendacji – użytkownik ocenił wszystkie produkty)\n");
        return;
    }
    for (int i = 0; i < show; i++)
        printf("  prod_%03d     %.4f\n", recs[i].product, recs[i].score);
}

/* =========================================================================
 * WCZYTYWANIE CSV w formacie: user_id,product_id,rating
 * (indeksy od 1; plik może zawierać nagłówek – automatyczne pominięcie)
 * ====================================================================== */
static int load_csv(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror(filename); return -1; }

    /* zerowanie macierzy ocen */
    memset(ratings, 0, sizeof(ratings));

    char line[MAX_LINE];
    int  line_no = 0;
    int  loaded  = 0;

    while (fgets(line, sizeof(line), f)) {
        line_no++;
        /* ewentualne usunięcie znaku nowej linii */
        line[strcspn(line, "\r\n")] = '\0';
        if (strlen(line) == 0) continue;

        /* pomiń nagłówek, jeśli zawiera "user" lub "User" */
        if (line_no == 1 && (strstr(line, "user") != NULL || strstr(line, "User") != NULL))
            continue;

        int uid, pid;
        float r;
        if (sscanf(line, "%d,%d,%f", &uid, &pid, &r) != 3) {
            fprintf(stderr, "Błąd formatu w linii %d: '%s'\n", line_no, line);
            continue;
        }
        if (uid < 1 || uid > MAX_USERS || pid < 1 || pid > MAX_PRODUCTS) {
            fprintf(stderr, "Pominięto linię %d: ID poza zakresem\n", line_no);
            continue;
        }
        ratings[uid-1][pid-1] = r;
        loaded++;
    }
    fclose(f);
    printf("Wczytano %d ocen (rzadka macierz %dx%d)\n", loaded, MAX_USERS, MAX_PRODUCTS);
    /* =========================================================================
 * DIAGNOSTYKA – podgląd fragmentu macierzy podobieństw
 * ====================================================================== */
static void print_sim_preview(float sim[MAX_USERS][MAX_USERS], int n) {
    if (n > num_users) n = num_users;
    printf("\n  Podglad macierzy podobienstwa (lewy gorny rog %dx%d):\n", n, n);
    printf("  %6s", "");
    for (int j = 0; j < n; j++) printf("  u%02d  ", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("  u%02d: ", i);
        for (int j = 0; j < n; j++) printf(" %5.3f", sim[i][j]);
        printf("\n");
    }
}

/* =========================================================================
 * MAIN
 * ====================================================================== */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uzycie: %s ratings.csv [user_id]\n", argv[0]);
        fprintf(stderr, "  user_id domyslnie: 0 (pierwszy uzytkownik)\n");
        return 1;
    }

    const char *csv_file   = argv[1];
    int         target_usr = (argc > 2) ? atoi(argv[2]) : 0;

    printf("\n=== Wczytywanie danych z %s ===\n", csv_file);
    if (load_csv(csv_file) < 0) return 1;

    if (target_usr < 0 || target_usr >= num_users) {
        fprintf(stderr, "Blad: user_id %d poza zakresem [0, %d]\n",
                target_usr, num_users - 1);
        return 1;
    }

    /* pre‑obliczenie norm */
    compute_norms();

    /* ---- Pomiar czasu dla różnej liczby wątków ---- */
    printf("\n=== Pomiar czasu budowania macierzy podobienstwa ===\n");
    printf("  %-10s  %-16s  %-12s\n", "Watki", "Czas [s]", "Przyspieszenie");
    printf("  %-10s  %-16s  %-12s\n", "----------", "----------------", "------------");

    double t_seq = build_similarity_sequential(sim_seq);
    printf("  %-10s  %-16.6f  %-12s\n", "sekw.", t_seq, "1.00x (bazowy)");

    int thread_counts[] = {1, 2, 4};
    for (int k = 0; k < 3; k++) {
        int nt = thread_counts[k];
        double t = build_similarity_parallel(nt, sim_par);
        double spd = t_seq / t;
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2fx", spd);
        printf("  %-10d  %-16.6f  %-12s\n", nt, t, buf);
    }

    /* Ostatni przebieg równoległy z 4 wątkami – używany dalej */
    build_similarity_parallel(4, sim_par);

    /* ---- Weryfikacja poprawności ---- */
    printf("\n=== Weryfikacja (rownolegle vs sekwencyjnie) ===\n");
    verify_results(sim_par, sim_seq);

    /* ---- Podgląd fragmentu macierzy ---- */
    print_sim_preview(sim_par, 5);

    /* ---- Rekomendacje dla wskazanego użytkownika ---- */
    int rated = 0;
    for (int p = 0; p < num_products; p++)
        if (ratings[target_usr][p] > 0.0f) rated++;
    printf("\n=== Top-%d rekomendacji dla uzytkownika %d ===\n",
           TOP_K, target_usr);
    printf("  Uzytkownik %d ocenil %d / %d produktow.\n",
           target_usr, rated, num_products);
    recommend_top_k(target_usr, sim_par);

    printf("\n=== Gotowe ===\n\n");
    return 0;
}
    return 0;
}
