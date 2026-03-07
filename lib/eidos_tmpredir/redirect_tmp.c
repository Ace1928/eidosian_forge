#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static char g_tmp_root[PATH_MAX];

static int (*real_open_fn)(const char *, int, ...) = NULL;
static int (*real_open64_fn)(const char *, int, ...) = NULL;
static int (*real_openat_fn)(int, const char *, int, ...) = NULL;
static int (*real_openat64_fn)(int, const char *, int, ...) = NULL;
static int (*real___open_2_fn)(const char *, int) = NULL;
static int (*real___openat_2_fn)(int, const char *, int) = NULL;
static FILE *(*real_fopen_fn)(const char *, const char *) = NULL;
static FILE *(*real_freopen_fn)(const char *, const char *, FILE *) = NULL;
static int (*real_access_fn)(const char *, int) = NULL;
static int (*real_stat_fn)(const char *, struct stat *) = NULL;
static int (*real_lstat_fn)(const char *, struct stat *) = NULL;
static int (*real_unlink_fn)(const char *) = NULL;
static int (*real_remove_fn)(const char *) = NULL;
static int (*real_rename_fn)(const char *, const char *) = NULL;
static int (*real_mkdir_fn)(const char *, mode_t) = NULL;
static DIR *(*real_opendir_fn)(const char *) = NULL;
static int (*real_mkstemp_fn)(char *) = NULL;
static int (*real_mkstemps_fn)(char *, int) = NULL;
static int (*real_mkostemp_fn)(char *, int) = NULL;
static int (*real_mkostemps_fn)(char *, int, int) = NULL;
static char *(*real_mkdtemp_fn)(char *) = NULL;

static void init_real_functions(void) {
    real_open_fn = dlsym(RTLD_NEXT, "open");
    real_open64_fn = dlsym(RTLD_NEXT, "open64");
    real_openat_fn = dlsym(RTLD_NEXT, "openat");
    real_openat64_fn = dlsym(RTLD_NEXT, "openat64");
    real___open_2_fn = dlsym(RTLD_NEXT, "__open_2");
    real___openat_2_fn = dlsym(RTLD_NEXT, "__openat_2");
    real_fopen_fn = dlsym(RTLD_NEXT, "fopen");
    real_freopen_fn = dlsym(RTLD_NEXT, "freopen");
    real_access_fn = dlsym(RTLD_NEXT, "access");
    real_stat_fn = dlsym(RTLD_NEXT, "stat");
    real_lstat_fn = dlsym(RTLD_NEXT, "lstat");
    real_unlink_fn = dlsym(RTLD_NEXT, "unlink");
    real_remove_fn = dlsym(RTLD_NEXT, "remove");
    real_rename_fn = dlsym(RTLD_NEXT, "rename");
    real_mkdir_fn = dlsym(RTLD_NEXT, "mkdir");
    real_opendir_fn = dlsym(RTLD_NEXT, "opendir");
    real_mkstemp_fn = dlsym(RTLD_NEXT, "mkstemp");
    real_mkstemps_fn = dlsym(RTLD_NEXT, "mkstemps");
    real_mkostemp_fn = dlsym(RTLD_NEXT, "mkostemp");
    real_mkostemps_fn = dlsym(RTLD_NEXT, "mkostemps");
    real_mkdtemp_fn = dlsym(RTLD_NEXT, "mkdtemp");
}

static bool path_starts_with(const char *path, const char *prefix) {
    size_t n = strlen(prefix);
    if (strncmp(path, prefix, n) != 0) {
        return false;
    }
    return path[n] == '\0' || path[n] == '/';
}

static bool should_redirect(const char *path) {
    if (!path || path[0] != '/') {
        return false;
    }
    return path_starts_with(path, "/tmp") || path_starts_with(path, "/var/tmp");
}

static int mkdir_p_internal(const char *path, mode_t mode) {
    char tmp[PATH_MAX];
    size_t len;
    char *p;

    if (!path || !*path) {
        return -1;
    }
    len = strnlen(path, sizeof(tmp) - 1);
    if (len == 0 || len >= sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(tmp, path, len);
    tmp[len] = '\0';

    for (p = tmp + 1; *p; ++p) {
        if (*p != '/') {
            continue;
        }
        *p = '\0';
        if (real_mkdir_fn && real_mkdir_fn(tmp, mode) != 0 && errno != EEXIST) {
            *p = '/';
            return -1;
        }
        *p = '/';
    }
    if (real_mkdir_fn && real_mkdir_fn(tmp, mode) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int ensure_parent_dir(const char *path, mode_t mode) {
    char parent[PATH_MAX];
    char *slash;
    size_t len;

    if (!path) {
        errno = EINVAL;
        return -1;
    }
    len = strnlen(path, sizeof(parent) - 1);
    if (len == 0 || len >= sizeof(parent)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(parent, path, len);
    parent[len] = '\0';
    slash = strrchr(parent, '/');
    if (!slash || slash == parent) {
        return 0;
    }
    *slash = '\0';
    return mkdir_p_internal(parent, mode);
}

static void init_tmp_root(void) {
    const char *tmp = getenv("EIDOS_TMPDIR");
    const char *prefix = getenv("PREFIX");
    const char *home = getenv("HOME");
    const char *user = getenv("USER");

    init_real_functions();
    if (tmp && *tmp) {
        snprintf(g_tmp_root, sizeof(g_tmp_root), "%s", tmp);
    } else if (prefix && strstr(prefix, "com.termux")) {
        snprintf(g_tmp_root, sizeof(g_tmp_root), "%s/tmp/eidos-%s", prefix, (user && *user) ? user : "termux");
    } else if (home && *home) {
        snprintf(g_tmp_root, sizeof(g_tmp_root), "%s/tmp", home);
    } else {
        snprintf(g_tmp_root, sizeof(g_tmp_root), "%s", "/tmp");
    }
    mkdir_p_internal(g_tmp_root, 0700);
}

static void ensure_initialized(void) {
    pthread_once(&g_init_once, init_tmp_root);
}

static const char *tmp_root(void) {
    ensure_initialized();
    return g_tmp_root;
}

static const char *redirect_path(const char *path, char *buffer, size_t size) {
    const char *root;
    const char *suffix;

    if (!should_redirect(path)) {
        return path;
    }
    root = tmp_root();
    suffix = path_starts_with(path, "/var/tmp") ? path + strlen("/var/tmp") : path + strlen("/tmp");
    if (*suffix == '\0') {
        snprintf(buffer, size, "%s", root);
    } else {
        snprintf(buffer, size, "%s%s", root, suffix);
    }
    return buffer;
}

int open(const char *pathname, int flags, ...) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(pathname, rewritten, sizeof(rewritten));
    va_list args;
    mode_t mode = 0;

    if ((flags & O_CREAT) != 0) {
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
        ensure_parent_dir(path, 0700);
        return real_open_fn(path, flags, mode);
    }
    return real_open_fn(path, flags);
}

int open64(const char *pathname, int flags, ...) {
    ensure_initialized();
    if (!real_open64_fn) {
        return open(pathname, flags);
    }
    char rewritten[PATH_MAX];
    const char *path = redirect_path(pathname, rewritten, sizeof(rewritten));
    va_list args;
    mode_t mode = 0;

    if ((flags & O_CREAT) != 0) {
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
        ensure_parent_dir(path, 0700);
        return real_open64_fn(path, flags, mode);
    }
    return real_open64_fn(path, flags);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    char rewritten[PATH_MAX];
    const char *path = pathname;
    ensure_initialized();
    va_list args;
    mode_t mode = 0;

    if (pathname && pathname[0] == '/') {
        path = redirect_path(pathname, rewritten, sizeof(rewritten));
    }
    if ((flags & O_CREAT) != 0) {
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
        ensure_parent_dir(path, 0700);
        return real_openat_fn(dirfd, path, flags, mode);
    }
    return real_openat_fn(dirfd, path, flags);
}

int openat64(int dirfd, const char *pathname, int flags, ...) {
    if (!real_openat64_fn) {
        return openat(dirfd, pathname, flags);
    }
    char rewritten[PATH_MAX];
    const char *path = pathname;
    va_list args;
    mode_t mode = 0;
    ensure_initialized();

    if (pathname && pathname[0] == '/') {
        path = redirect_path(pathname, rewritten, sizeof(rewritten));
    }
    if ((flags & O_CREAT) != 0) {
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
        ensure_parent_dir(path, 0700);
        return real_openat64_fn(dirfd, path, flags, mode);
    }
    return real_openat64_fn(dirfd, path, flags);
}

int __open_2(const char *pathname, int flags) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real___open_2_fn
        ? real___open_2_fn(redirect_path(pathname, rewritten, sizeof(rewritten)), flags)
        : open(pathname, flags);
}

int __openat_2(int dirfd, const char *pathname, int flags) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    if (real___openat_2_fn) {
        if (pathname && pathname[0] == '/') {
            pathname = redirect_path(pathname, rewritten, sizeof(rewritten));
        }
        return real___openat_2_fn(dirfd, pathname, flags);
    }
    return openat(dirfd, pathname, flags);
}

FILE *fopen(const char *pathname, const char *mode) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(pathname, rewritten, sizeof(rewritten));
    if (mode && (strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+'))) {
        ensure_parent_dir(path, 0700);
    }
    return real_fopen_fn(path, mode);
}

FILE *freopen(const char *pathname, const char *mode, FILE *stream) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(pathname, rewritten, sizeof(rewritten));
    if (mode && (strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+'))) {
        ensure_parent_dir(path, 0700);
    }
    return real_freopen_fn(path, mode, stream);
}

int access(const char *pathname, int mode) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_access_fn(redirect_path(pathname, rewritten, sizeof(rewritten)), mode);
}

int stat(const char *pathname, struct stat *statbuf) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_stat_fn(redirect_path(pathname, rewritten, sizeof(rewritten)), statbuf);
}

int lstat(const char *pathname, struct stat *statbuf) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_lstat_fn(redirect_path(pathname, rewritten, sizeof(rewritten)), statbuf);
}

int unlink(const char *pathname) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_unlink_fn(redirect_path(pathname, rewritten, sizeof(rewritten)));
}

int remove(const char *pathname) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_remove_fn(redirect_path(pathname, rewritten, sizeof(rewritten)));
}

int rename(const char *oldpath, const char *newpath) {
    char old_rewritten[PATH_MAX];
    char new_rewritten[PATH_MAX];
    ensure_initialized();
    const char *old_real = redirect_path(oldpath, old_rewritten, sizeof(old_rewritten));
    const char *new_real = redirect_path(newpath, new_rewritten, sizeof(new_rewritten));
    ensure_parent_dir(new_real, 0700);
    return real_rename_fn(old_real, new_real);
}

int mkdir(const char *pathname, mode_t mode) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(pathname, rewritten, sizeof(rewritten));
    ensure_parent_dir(path, 0700);
    return real_mkdir_fn(path, mode);
}

DIR *opendir(const char *name) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    return real_opendir_fn(redirect_path(name, rewritten, sizeof(rewritten)));
}

int mkstemp(char *template) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(template, rewritten, sizeof(rewritten));
    if (path != template) {
        snprintf(template, PATH_MAX, "%s", path);
    }
    ensure_parent_dir(template, 0700);
    return real_mkstemp_fn(template);
}

int mkstemps(char *template, int suffixlen) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(template, rewritten, sizeof(rewritten));
    if (path != template) {
        snprintf(template, PATH_MAX, "%s", path);
    }
    ensure_parent_dir(template, 0700);
    return real_mkstemps_fn(template, suffixlen);
}

int mkostemp(char *template, int flags) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(template, rewritten, sizeof(rewritten));
    if (path != template) {
        snprintf(template, PATH_MAX, "%s", path);
    }
    ensure_parent_dir(template, 0700);
    return real_mkostemp_fn(template, flags);
}

int mkostemps(char *template, int suffixlen, int flags) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(template, rewritten, sizeof(rewritten));
    if (path != template) {
        snprintf(template, PATH_MAX, "%s", path);
    }
    ensure_parent_dir(template, 0700);
    return real_mkostemps_fn(template, suffixlen, flags);
}

char *mkdtemp(char *template) {
    char rewritten[PATH_MAX];
    ensure_initialized();
    const char *path = redirect_path(template, rewritten, sizeof(rewritten));
    if (path != template) {
        snprintf(template, PATH_MAX, "%s", path);
    }
    ensure_parent_dir(template, 0700);
    return real_mkdtemp_fn(template);
}
