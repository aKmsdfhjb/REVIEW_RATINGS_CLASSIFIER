# Review Ratings Classifier — Architecture

## System flow

```mermaid
flowchart TD
 A[Review text and optional rating] --> B{Selected model}
 B -->|Ensemble| C[TF-IDF text pipeline]
 B -->|SVM| D[TF-IDF plus rating preprocessor]
 C --> E[Voting classifier]
 D --> F[Linear SVM]
 E --> G[Predicted label]
 F --> G
 G --> H[Flask JSON response]
 H --> I[Browser result panel]
```

## Design notes

- Keep user-interface responsibilities separate from model, API and data-service responsibilities.
- Validate inputs at the application boundary and return clear error messages.
- Keep secrets and privileged credentials on the server side.
- Document the versions of data, dependencies and model artifacts used for reproducible results.
- Test the full user journey after deployment, not only the build step.

## Deployment boundary

GitHub Pages can host static HTML, CSS, JavaScript and prebuilt frontend assets. It cannot execute Python/Flask services, server-side React Router actions, private environment variables or database operations. Those parts require a separate backend or application host.
